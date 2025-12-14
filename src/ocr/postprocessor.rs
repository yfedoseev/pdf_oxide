//! DBNet++ postprocessing for text detection.
//!
//! This module converts the probability map output from DBNet++ into
//! text bounding boxes through binarization, connected components analysis,
//! and polygon extraction.

use ndarray::{Array2, ArrayView2};

use super::error::{OcrError, OcrResult};

/// A detected text box with quadrilateral coordinates and confidence.
#[derive(Debug, Clone)]
pub struct DetectedBox {
    /// Four corner points of the text box [top-left, top-right, bottom-right, bottom-left]
    pub polygon: [[f32; 2]; 4],
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Extract text boxes from DBNet++ probability map.
///
/// # Arguments
///
/// * `prob_map` - 2D probability map from detector (H x W)
/// * `threshold` - Binarization threshold (typically 0.3)
/// * `box_threshold` - Minimum confidence to keep a box (typically 0.5)
/// * `max_candidates` - Maximum number of boxes to return
/// * `unclip_ratio` - Expansion ratio for boxes (typically 1.5)
/// * `scale` - Scale factor to convert back to original image coordinates
///
/// # Returns
///
/// Vector of detected text boxes.
pub fn extract_boxes(
    prob_map: ArrayView2<f32>,
    threshold: f32,
    box_threshold: f32,
    max_candidates: usize,
    unclip_ratio: f32,
    scale: f32,
) -> OcrResult<Vec<DetectedBox>> {
    let (height, width) = prob_map.dim();

    if height == 0 || width == 0 {
        return Err(OcrError::PostprocessingError("Empty probability map".to_string()));
    }

    // Step 1: Binarize the probability map
    let binary = binarize(prob_map, threshold);

    // Step 2: Find connected components (contours)
    let contours = find_contours(&binary);

    // Step 3: Process each contour into a bounding box
    let mut boxes = Vec::new();

    for contour in contours.into_iter().take(max_candidates) {
        if contour.len() < 4 {
            continue;
        }

        // Calculate contour score (mean probability inside)
        let score = calculate_contour_score(prob_map, &contour);
        if score < box_threshold {
            continue;
        }

        // Get minimum bounding box
        let min_rect = min_area_rect(&contour);

        // Expand (unclip) the box
        let expanded = unclip_polygon(&min_rect, unclip_ratio);

        // Scale back to original image coordinates
        let scaled: [[f32; 2]; 4] = [
            [expanded[0][0] / scale, expanded[0][1] / scale],
            [expanded[1][0] / scale, expanded[1][1] / scale],
            [expanded[2][0] / scale, expanded[2][1] / scale],
            [expanded[3][0] / scale, expanded[3][1] / scale],
        ];

        boxes.push(DetectedBox {
            polygon: scaled,
            confidence: score,
        });
    }

    // Sort by confidence (highest first)
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(boxes)
}

/// Binarize probability map using threshold.
fn binarize(prob_map: ArrayView2<f32>, threshold: f32) -> Array2<bool> {
    prob_map.mapv(|p| p > threshold)
}

/// Simple connected components analysis using flood fill.
///
/// Returns a vector of contours, where each contour is a vector of (x, y) points.
fn find_contours(binary: &Array2<bool>) -> Vec<Vec<[usize; 2]>> {
    let (height, width) = binary.dim();
    let mut visited = Array2::<bool>::default((height, width));
    let mut contours = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if binary[[y, x]] && !visited[[y, x]] {
                // Found a new component - flood fill to get boundary
                let contour = flood_fill_boundary(binary, &mut visited, x, y);
                if !contour.is_empty() {
                    contours.push(contour);
                }
            }
        }
    }

    contours
}

/// Flood fill to find boundary points of a connected component.
fn flood_fill_boundary(
    binary: &Array2<bool>,
    visited: &mut Array2<bool>,
    start_x: usize,
    start_y: usize,
) -> Vec<[usize; 2]> {
    let (height, width) = binary.dim();
    let mut stack = vec![(start_x, start_y)];
    let mut boundary_points = Vec::new();
    let mut min_x = start_x;
    let mut max_x = start_x;
    let mut min_y = start_y;
    let mut max_y = start_y;

    // 4-connectivity directions
    let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    while let Some((x, y)) = stack.pop() {
        if visited[[y, x]] {
            continue;
        }
        visited[[y, x]] = true;

        // Track bounding box
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);

        // Check if this is a boundary pixel
        let mut is_boundary = false;
        for (dx, dy) in &directions {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;

            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                is_boundary = true;
            } else {
                let (nx, ny) = (nx as usize, ny as usize);
                if !binary[[ny, nx]] {
                    is_boundary = true;
                } else if !visited[[ny, nx]] {
                    stack.push((nx, ny));
                }
            }
        }

        if is_boundary {
            boundary_points.push([x, y]);
        }
    }

    // If we have a valid region, return simplified boundary
    if max_x > min_x && max_y > min_y {
        // For simplicity, we'll return the bounding box corners
        // A more sophisticated implementation would trace the actual contour
        boundary_points
    } else {
        Vec::new()
    }
}

/// Calculate the mean probability score inside a contour.
fn calculate_contour_score(prob_map: ArrayView2<f32>, contour: &[[usize; 2]]) -> f32 {
    if contour.is_empty() {
        return 0.0;
    }

    // Find bounding box of contour
    let min_x = contour.iter().map(|p| p[0]).min().unwrap_or(0);
    let max_x = contour.iter().map(|p| p[0]).max().unwrap_or(0);
    let min_y = contour.iter().map(|p| p[1]).min().unwrap_or(0);
    let max_y = contour.iter().map(|p| p[1]).max().unwrap_or(0);

    // Calculate mean probability in bounding box
    let mut sum = 0.0;
    let mut count = 0;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if y < prob_map.dim().0 && x < prob_map.dim().1 {
                sum += prob_map[[y, x]];
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

/// Get minimum area bounding rectangle from contour points.
fn min_area_rect(contour: &[[usize; 2]]) -> [[f32; 2]; 4] {
    if contour.is_empty() {
        return [[0.0; 2]; 4];
    }

    // Find axis-aligned bounding box
    let min_x = contour.iter().map(|p| p[0]).min().unwrap_or(0) as f32;
    let max_x = contour.iter().map(|p| p[0]).max().unwrap_or(0) as f32;
    let min_y = contour.iter().map(|p| p[1]).min().unwrap_or(0) as f32;
    let max_y = contour.iter().map(|p| p[1]).max().unwrap_or(0) as f32;

    // Return as quadrilateral: top-left, top-right, bottom-right, bottom-left
    [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ]
}

/// Expand polygon using unclip ratio.
///
/// Uses simplified expansion: moves each edge outward by a distance
/// proportional to the box size.
fn unclip_polygon(polygon: &[[f32; 2]; 4], ratio: f32) -> [[f32; 2]; 4] {
    // Calculate center
    let cx: f32 = polygon.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cy: f32 = polygon.iter().map(|p| p[1]).sum::<f32>() / 4.0;

    // Expand each point away from center
    let expansion = (ratio - 1.0) / 2.0;

    let mut expanded = [[0.0f32; 2]; 4];
    for (i, point) in polygon.iter().enumerate() {
        let dx = point[0] - cx;
        let dy = point[1] - cy;
        expanded[i][0] = point[0] + dx * expansion;
        expanded[i][1] = point[1] + dy * expansion;
    }

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_binarize() {
        let prob_map =
            Array2::from_shape_vec((3, 3), vec![0.1, 0.5, 0.9, 0.2, 0.6, 0.8, 0.3, 0.4, 0.7])
                .unwrap();

        let binary = binarize(prob_map.view(), 0.5);

        assert!(!binary[[0, 0]]); // 0.1 < 0.5
        assert!(!binary[[0, 1]]); // 0.5 not > 0.5
        assert!(binary[[0, 2]]); // 0.9 > 0.5
        assert!(binary[[1, 1]]); // 0.6 > 0.5
    }

    #[test]
    fn test_min_area_rect() {
        let contour = vec![[10, 20], [50, 20], [50, 40], [10, 40]];
        let rect = min_area_rect(&contour);

        assert!((rect[0][0] - 10.0).abs() < f32::EPSILON);
        assert!((rect[0][1] - 20.0).abs() < f32::EPSILON);
        assert!((rect[2][0] - 50.0).abs() < f32::EPSILON);
        assert!((rect[2][1] - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_unclip_polygon() {
        let polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        let expanded = unclip_polygon(&polygon, 1.5);

        // Center is (50, 25)
        // With ratio 1.5, expansion factor is 0.25
        // Top-left (0, 0) -> (0 + (0-50)*0.25, 0 + (0-25)*0.25) = (-12.5, -6.25)
        assert!(expanded[0][0] < 0.0); // Expanded left
        assert!(expanded[0][1] < 0.0); // Expanded up
        assert!(expanded[2][0] > 100.0); // Expanded right
        assert!(expanded[2][1] > 50.0); // Expanded down
    }

    #[test]
    fn test_extract_boxes_empty() {
        let prob_map = Array2::<f32>::zeros((100, 100));
        let boxes = extract_boxes(prob_map.view(), 0.3, 0.5, 100, 1.5, 1.0).unwrap();
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_extract_boxes_single_region() {
        // Create a probability map with a high-probability region
        let mut prob_map = Array2::<f32>::zeros((100, 100));
        for y in 20..40 {
            for x in 30..70 {
                prob_map[[y, x]] = 0.9;
            }
        }

        let boxes = extract_boxes(prob_map.view(), 0.3, 0.5, 100, 1.5, 1.0).unwrap();
        assert!(!boxes.is_empty());

        // Check that the box roughly covers the high-probability region
        let box0 = &boxes[0];
        assert!(box0.confidence > 0.5);
    }
}
