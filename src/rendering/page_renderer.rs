//! Page renderer using tiny-skia.
//!
//! This module implements the core PDF rendering logic, converting
//! PDF operators into tiny-skia drawing commands.
#![allow(
    clippy::manual_div_ceil,
    clippy::field_reassign_with_default,
    clippy::collapsible_if,
    clippy::needless_borrow,
    clippy::get_first,
    clippy::if_same_then_else,
    clippy::needless_return_with_question_mark,
    clippy::ptr_arg
)]

use crate::content::graphics_state::{GraphicsState, GraphicsStateStack, Matrix};
use crate::content::operators::Operator;
use crate::content::parser::parse_content_stream;
use crate::document::PdfDocument;
use crate::error::{Error, Result};
use crate::object::{Object, ObjectRef};
use crate::rendering::path_rasterizer::PathRasterizer;
use crate::rendering::text_rasterizer::TextRasterizer;

use crate::fonts::FontInfo;
use std::collections::HashMap;
use std::sync::Arc;
use tiny_skia::{Color, PathBuilder, Pixmap, PixmapPaint, Transform};

/// Image output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// Portable Network Graphics
    Png,
    /// Joint Photographic Experts Group
    Jpeg,
}

/// Options for page rendering.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Resolution in dots per inch (default: 150)
    pub dpi: u32,
    /// Output image format (default: PNG)
    pub format: ImageFormat,
    /// Background color (RGBA, default: white)
    pub background: Option<[f32; 4]>,
    /// Whether to render annotations (default: true)
    pub render_annotations: bool,
    /// JPEG quality (1-100, default: 85)
    pub jpeg_quality: u8,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            dpi: 150,
            format: ImageFormat::Png,
            background: Some([1.0, 1.0, 1.0, 1.0]), // White background
            render_annotations: true,
            jpeg_quality: 85,
        }
    }
}

impl RenderOptions {
    /// Set a transparent background (no background fill).
    pub fn with_transparent_background(mut self) -> Self {
        self.background = None;
        self
    }
}

impl RenderOptions {
    /// Create options with specified DPI.
    pub fn with_dpi(dpi: u32) -> Self {
        Self {
            dpi,
            ..Default::default()
        }
    }

    /// Set format to JPEG with quality (clamped to 1-100).
    pub fn as_jpeg(mut self, quality: u8) -> Self {
        self.format = ImageFormat::Jpeg;
        self.jpeg_quality = quality.clamp(1, 100);
        self
    }
}

/// A rendered page image.
pub struct RenderedImage {
    /// Raw image data
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Format of the image data
    pub format: ImageFormat,
}

impl RenderedImage {
    /// Save the image to a file.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, &self.data)
            .map_err(|e| Error::InvalidPdf(format!("Failed to write image: {}", e)))
    }

    /// Get the image data as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Page renderer that converts PDF pages to raster images.
pub struct PageRenderer {
    options: RenderOptions,
    path_rasterizer: PathRasterizer,
    text_rasterizer: TextRasterizer,
    /// Font cache (name -> FontInfo) for current context
    fonts: HashMap<String, Arc<FontInfo>>,
    /// Color space cache (name -> Object) for current context
    color_spaces: HashMap<String, Object>,
}

impl PageRenderer {
    /// Create a new page renderer with the specified options.
    pub fn new(options: RenderOptions) -> Self {
        Self {
            options,
            path_rasterizer: PathRasterizer::new(),
            text_rasterizer: TextRasterizer::new(),
            fonts: HashMap::new(),
            color_spaces: HashMap::new(),
        }
    }

    /// Render a page to a raster image.
    pub fn render_page(&mut self, doc: &mut PdfDocument, page_num: usize) -> Result<RenderedImage> {
        self.render_page_with_options(page_num, doc)
    }

    /// Render a page with specific options.
    pub fn render_page_with_options(
        &mut self,
        page_num: usize,
        doc: &mut PdfDocument,
    ) -> Result<RenderedImage> {
        // Clear caches for new page
        self.fonts.clear();
        self.color_spaces.clear();

        // Get page info
        let page_info = doc.get_page_info(page_num)?;
        let media_box = page_info.media_box;

        // Calculate output dimensions, accounting for page rotation
        let scale = self.options.dpi as f32 / 72.0;
        let rotation = page_info.rotation % 360;
        let (page_w, page_h) = if rotation == 90 || rotation == 270 {
            (media_box.height, media_box.width) // Swap for landscape
        } else {
            (media_box.width, media_box.height)
        };
        let width = (page_w * scale).ceil() as u32;
        let height = (page_h * scale).ceil() as u32;

        // Create pixmap
        let mut pixmap = Pixmap::new(width, height)
            .ok_or_else(|| Error::InvalidPdf("Failed to create pixmap".to_string()))?;

        // Fill background
        if let Some(bg) = self.options.background {
            let [r, g, b, a] = bg;
            pixmap.fill(Color::from_rgba(r, g, b, a).unwrap_or(Color::WHITE));
        }

        // Create base transform: PDF coordinates to pixel coordinates
        // PDF origin is bottom-left; we flip Y and apply page rotation.
        // Per PDF spec §8.3.2.3, /Rotate specifies clockwise rotation.
        // The approach: first map PDF coords to an unrotated pixel space,
        // then rotate the entire result.
        let transform = match rotation {
            90 => {
                // 90° CW rotation: portrait PDF → landscape display
                // PDF y-up (x,y) → screen y-down: screen_x = y*s, screen_y = x*s
                Transform::from_translate(-media_box.x, -media_box.y)
                    .post_concat(Transform::from_row(0.0, scale, scale, 0.0, 0.0, 0.0))
            },
            180 => Transform::from_translate(-media_box.x, -media_box.y)
                .post_scale(-scale, scale)
                .post_translate(media_box.width * scale, 0.0),
            270 => Transform::from_translate(-media_box.x, -media_box.y).post_concat(
                Transform::from_row(0.0, scale, -scale, 0.0, media_box.height * scale, 0.0),
            ),
            _ => {
                // No rotation (0°)
                Transform::from_translate(-media_box.x, -media_box.y)
                    .post_scale(scale, -scale)
                    .post_translate(0.0, page_h * scale)
            },
        };

        // Get page resources
        let resources = doc.get_page_resources(page_num)?;

        // Pre-load resources (v0.3.18 synchronization)
        self.load_resources(doc, &resources)?;

        // Get page content stream
        let content_data = doc.get_page_content_data(page_num)?;

        // Parse content stream
        let operators = match parse_content_stream(&content_data) {
            Ok(ops) => ops,
            Err(e) => {
                return Err(e);
            },
        };

        // Execute operators
        self.execute_operators(&mut pixmap, transform, &operators, doc, page_num, &resources)?;

        // Render annotations (if requested and present)
        if self.options.render_annotations {
            self.render_annotations(&mut pixmap, transform, doc, page_num)?;
        }

        // Encode to output format
        let data = match self.options.format {
            ImageFormat::Png => pixmap
                .encode_png()
                .map_err(|e| Error::InvalidPdf(format!("PNG encoding failed: {}", e)))?,
            ImageFormat::Jpeg => self.encode_jpeg(&pixmap)?,
        };

        Ok(RenderedImage {
            data,
            width,
            height,
            format: self.options.format,
        })
    }

    /// Load resources (fonts, color spaces) into local cache.
    fn load_resources(&mut self, doc: &mut PdfDocument, resources: &Object) -> Result<()> {
        if let Object::Dictionary(res_dict) = resources {
            log::debug!("Loading resources, keys: {:?}", res_dict.keys());
            // Fonts
            if let Some(font_obj) = res_dict.get("Font") {
                log::debug!("Found Font resource");
                let font_dict_obj = doc.resolve_object(font_obj)?;
                if let Some(font_dict) = font_dict_obj.as_dict() {
                    for (name, f_obj) in font_dict {
                        let resolved_f = doc.resolve_object(f_obj)?;
                        if let Ok(info) = FontInfo::from_dict(&resolved_f, doc) {
                            log::debug!("Resolved font '{}': subtype={}, encoding={:?}, has_to_unicode={}, has_embedded={}", 
                                info.base_font, info.subtype, info.encoding, info.to_unicode.is_some(), info.embedded_font_data.is_some());
                            self.fonts.insert(name.clone(), Arc::new(info));
                        }
                    }
                }
            }

            // Color Spaces
            if let Some(cs_obj) = res_dict.get("ColorSpace") {
                log::debug!("Found ColorSpace resource");
                let cs_dict_obj = doc.resolve_object(cs_obj)?;
                if let Some(cs_dict) = cs_dict_obj.as_dict() {
                    for (name, o) in cs_dict {
                        if let Ok(resolved_cs) = doc.resolve_object(o) {
                            log::debug!("Resolved color space '{}': {:?}", name, resolved_cs);
                            self.color_spaces.insert(name.clone(), resolved_cs);
                        }
                    }
                }
            }

            // XObjects
            if let Some(xobj_obj) = res_dict.get("XObject") {
                let xobj_dict_obj = doc.resolve_object(xobj_obj)?;
                if let Some(xobj_dict) = xobj_dict_obj.as_dict() {
                    log::debug!("XObject dict keys: {:?}", xobj_dict.keys());
                }
            }
        }

        // Share TrueType CMaps between matching fonts (essential for CID fonts with missing ToUnicode)
        self.share_truetype_cmaps();
        Ok(())
    }

    /// Share TrueType cmap tables between fonts with matching base font names.
    fn share_truetype_cmaps(&mut self) {
        let mut base_font_to_cmap = HashMap::new();

        // First pass: collect available cmaps
        for font in self.fonts.values() {
            if let Some(cmap) = font.truetype_cmap() {
                // Get base font name without subset prefix (e.g. ABCDEF+Arial -> Arial)
                let base_name = if let Some(plus_idx) = font.base_font.find('+') {
                    &font.base_font[plus_idx + 1..]
                } else {
                    &font.base_font
                };
                base_font_to_cmap.insert(base_name.to_string(), cmap.clone());
            }
        }

        // Second pass: apply cmaps to fonts missing them
        for font in self.fonts.values() {
            if font.subtype == "Type0" && font.truetype_cmap().is_none() {
                let base_name = if let Some(plus_idx) = font.base_font.find('+') {
                    &font.base_font[plus_idx + 1..]
                } else {
                    &font.base_font
                };
                if let Some(shared_cmap) = base_font_to_cmap.get(base_name) {
                    font.truetype_cmap.set(Some(shared_cmap.clone())).ok();
                }
            }
        }
    }

    /// Execute PDF operators to render content.
    fn execute_operators(
        &mut self,
        pixmap: &mut Pixmap,
        base_transform: Transform,
        operators: &[Operator],
        doc: &mut PdfDocument,
        page_num: usize,
        resources: &Object,
    ) -> Result<()> {
        let mut gs_stack = GraphicsStateStack::new();

        // PDF default: DeviceGray, black
        {
            let gs = gs_stack.current_mut();
            gs.fill_color_space = "DeviceGray".to_string();
            gs.stroke_color_space = "DeviceGray".to_string();
            gs.fill_color_rgb = (0.0, 0.0, 0.0);
            gs.stroke_color_rgb = (0.0, 0.0, 0.0);
        }

        let mut in_text_object = false;
        let mut current_path = PathBuilder::new();
        let mut pending_clip: Option<(tiny_skia::Path, tiny_skia::FillRule)> = None;
        let mut clip_stack: Vec<Option<tiny_skia::Mask>> = vec![None]; // Start with no clip at depth 0

        for op in operators {
            match op {
                // Graphics state operators
                Operator::SaveState => {
                    gs_stack.save();
                    // Clone current clip for the new graphics state level
                    // This allows the current level to modify its clip without affecting parents
                    let current_clip = clip_stack.last().cloned().flatten();
                    clip_stack.push(current_clip);
                    log::debug!(
                        "q (SaveState), depth={}, clip_stack depth={}",
                        gs_stack.depth(),
                        clip_stack.len()
                    );
                },
                Operator::RestoreState => {
                    gs_stack.restore();
                    // Restore previous clipping region by popping current level
                    if clip_stack.len() > 1 {
                        clip_stack.pop();
                    }
                    log::debug!(
                        "Q (RestoreState), depth={}, clip_stack depth={}",
                        gs_stack.depth(),
                        clip_stack.len()
                    );
                },
                Operator::Cm { a, b, c, d, e, f } => {
                    let matrix = Matrix {
                        a: *a,
                        b: *b,
                        c: *c,
                        d: *d,
                        e: *e,
                        f: *f,
                    };
                    let current = gs_stack.current_mut();
                    // PDF spec ISO 32000-1:2008 §8.3.4: cm concatenates as M_cm × CTM
                    current.ctm = matrix.multiply(&current.ctm);
                    log::debug!(
                        "cm: [{}, {}, {}, {}, {}, {}], CTM now: {:?}",
                        a,
                        b,
                        c,
                        d,
                        e,
                        f,
                        current.ctm
                    );
                },

                // Color operators
                Operator::SetFillRgb { r, g, b } => {
                    gs_stack.current_mut().fill_color_rgb = (*r, *g, *b);
                    gs_stack.current_mut().fill_color_space = "DeviceRGB".to_string();
                    log::debug!("SetFillRgb: [{}, {}, {}]", r, g, b);
                },
                Operator::SetStrokeRgb { r, g, b } => {
                    gs_stack.current_mut().stroke_color_rgb = (*r, *g, *b);
                    gs_stack.current_mut().stroke_color_space = "DeviceRGB".to_string();
                    log::debug!("SetStrokeRgb: [{}, {}, {}]", r, g, b);
                },
                Operator::SetFillGray { gray } => {
                    let g = *gray;
                    gs_stack.current_mut().fill_color_rgb = (g, g, g);
                    gs_stack.current_mut().fill_color_space = "DeviceGray".to_string();
                    log::debug!("SetFillGray: {}", g);
                },
                Operator::SetStrokeGray { gray } => {
                    let g = *gray;
                    gs_stack.current_mut().stroke_color_rgb = (g, g, g);
                    gs_stack.current_mut().stroke_color_space = "DeviceGray".to_string();
                    log::debug!("SetStrokeGray: {}", g);
                },
                Operator::SetFillCmyk { c, m, y, k } => {
                    // Convert CMYK to RGB
                    let (r, g, b) = cmyk_to_rgb(*c, *m, *y, *k);
                    gs_stack.current_mut().fill_color_rgb = (r, g, b);
                    gs_stack.current_mut().fill_color_cmyk = Some((*c, *m, *y, *k));
                    gs_stack.current_mut().fill_color_space = "DeviceCMYK".to_string();
                    log::debug!("SetFillCmyk: [{}, {}, {}, {}] -> {:?}", c, m, y, k, (r, g, b));
                },
                Operator::SetStrokeCmyk { c, m, y, k } => {
                    let (r, g, b) = cmyk_to_rgb(*c, *m, *y, *k);
                    gs_stack.current_mut().stroke_color_rgb = (r, g, b);
                    gs_stack.current_mut().stroke_color_cmyk = Some((*c, *m, *y, *k));
                    gs_stack.current_mut().stroke_color_space = "DeviceCMYK".to_string();
                    log::debug!("SetStrokeCmyk: [{}, {}, {}, {}] -> {:?}", c, m, y, k, (r, g, b));
                },

                // Color space operators
                Operator::SetFillColorSpace { name } => {
                    gs_stack.current_mut().fill_color_space = name.clone();
                    log::debug!("SetFillColorSpace: {}", name);
                },
                Operator::SetStrokeColorSpace { name } => {
                    gs_stack.current_mut().stroke_color_space = name.clone();
                },
                Operator::SetFillColor { components } => {
                    let gs = gs_stack.current_mut();
                    let space_name = gs.fill_color_space.clone();
                    let resolved_space = self.color_spaces.get(&space_name);

                    match space_name.as_str() {
                        "DeviceGray" | "G" if !components.is_empty() => {
                            let g = components[0];
                            gs.fill_color_rgb = (g, g, g);
                        },
                        "DeviceRGB" | "RGB" if components.len() >= 3 => {
                            gs.fill_color_rgb = (components[0], components[1], components[2]);
                        },
                        "DeviceCMYK" | "CMYK" if components.len() >= 4 => {
                            gs.fill_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        },
                        _ => {
                            let mut handled = false;
                            if let Some(rs) = resolved_space {
                                if let Some(arr) = rs.as_array() {
                                    if let Some(type_name) = arr.first().and_then(|o| o.as_name()) {
                                        match type_name {
                                            "ICCBased" if arr.len() > 1 => {
                                                if let Ok(dict_obj) = doc.resolve_object(&arr[1]) {
                                                    if let Some(dict) = dict_obj.as_dict() {
                                                        let n = dict
                                                            .get("N")
                                                            .and_then(|o| o.as_integer())
                                                            .unwrap_or(3);
                                                        match n {
                                                            1 if !components.is_empty() => {
                                                                let g = components[0];
                                                                gs.fill_color_rgb = (g, g, g);
                                                                handled = true;
                                                            },
                                                            3 if components.len() >= 3 => {
                                                                gs.fill_color_rgb = (
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                );
                                                                handled = true;
                                                            },
                                                            4 if components.len() >= 4 => {
                                                                gs.fill_color_rgb = cmyk_to_rgb(
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                    components[3],
                                                                );
                                                                handled = true;
                                                            },
                                                            _ => {},
                                                        }
                                                    }
                                                }
                                            },
                                            "Separation" | "DeviceN" => {
                                                // Per PDF spec, Separation = [/Separation name altCS tintTransform]
                                                // Evaluate tint transform against alternate color space
                                                if !components.is_empty() {
                                                    let tint = components[0];
                                                    let alt_cs = arr
                                                        .get(2)
                                                        .and_then(|o| o.as_name())
                                                        .unwrap_or("");
                                                    if alt_cs == "DeviceCMYK" && arr.len() >= 4 {
                                                        if let Some(func_obj) = arr.get(3) {
                                                            if let Ok(func_res) =
                                                                doc.resolve_object(func_obj)
                                                            {
                                                                if let Some(fd) = func_res.as_dict()
                                                                {
                                                                    if fd
                                                                        .get("FunctionType")
                                                                        .and_then(|o| {
                                                                            o.as_integer()
                                                                        })
                                                                        == Some(2)
                                                                    {
                                                                        let c0 =
                                                                            fd.get("C0").and_then(
                                                                                |o| o.as_array(),
                                                                            );
                                                                        let c1 =
                                                                            fd.get("C1").and_then(
                                                                                |o| o.as_array(),
                                                                            );
                                                                        let get_f = |arr: Option<&Vec<Object>>, i: usize, def: f32| -> f32 {
                                                                            arr.and_then(|a| a.get(i)).map(|o| match o { Object::Real(v) => *v as f32, Object::Integer(v) => *v as f32, _ => def }).unwrap_or(def)
                                                                        };
                                                                        let c = get_f(c0, 0, 0.0)
                                                                            + tint
                                                                                * (get_f(
                                                                                    c1, 0, 0.0,
                                                                                ) - get_f(
                                                                                    c0, 0, 0.0,
                                                                                ));
                                                                        let m = get_f(c0, 1, 0.0)
                                                                            + tint
                                                                                * (get_f(
                                                                                    c1, 1, 0.0,
                                                                                ) - get_f(
                                                                                    c0, 1, 0.0,
                                                                                ));
                                                                        let y = get_f(c0, 2, 0.0)
                                                                            + tint
                                                                                * (get_f(
                                                                                    c1, 2, 0.0,
                                                                                ) - get_f(
                                                                                    c0, 2, 0.0,
                                                                                ));
                                                                        let k = get_f(c0, 3, 0.0)
                                                                            + tint
                                                                                * (get_f(
                                                                                    c1, 3, 1.0,
                                                                                ) - get_f(
                                                                                    c0, 3, 0.0,
                                                                                ));
                                                                        gs.fill_color_rgb =
                                                                            cmyk_to_rgb(c, m, y, k);
                                                                        handled = true;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    if !handled {
                                                        let g = 1.0 - tint;
                                                        gs.fill_color_rgb = (g, g, g);
                                                    }
                                                    handled = true;
                                                }
                                            },
                                            "Indexed" => {
                                                if !components.is_empty() {
                                                    let g = components[0] / 255.0;
                                                    gs.fill_color_rgb = (g, g, g);
                                                    handled = true;
                                                }
                                            },
                                            _ => {},
                                        }
                                    }
                                }
                            }

                            if !handled && !components.is_empty() {
                                let g = components[0];
                                gs.fill_color_rgb = (g, g, g);
                            }
                        },
                    }
                    log::debug!(
                        "SetFillColor: {} {:?} -> {:?}",
                        space_name,
                        components,
                        gs.fill_color_rgb
                    );
                },
                Operator::SetStrokeColor { components } => {
                    let gs = gs_stack.current_mut();
                    let space_name = gs.stroke_color_space.clone();
                    let resolved_space = self.color_spaces.get(&space_name);

                    match space_name.as_str() {
                        "DeviceGray" | "G" if !components.is_empty() => {
                            let g = components[0];
                            gs.stroke_color_rgb = (g, g, g);
                        },
                        "DeviceRGB" | "RGB" if components.len() >= 3 => {
                            gs.stroke_color_rgb = (components[0], components[1], components[2]);
                        },
                        "DeviceCMYK" | "CMYK" if components.len() >= 4 => {
                            gs.stroke_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        },
                        _ => {
                            let mut handled = false;
                            if let Some(rs) = resolved_space {
                                if let Some(arr) = rs.as_array() {
                                    if let Some(type_name) = arr.first().and_then(|o| o.as_name()) {
                                        match type_name {
                                            "ICCBased" if arr.len() > 1 => {
                                                if let Ok(dict_obj) = doc.resolve_object(&arr[1]) {
                                                    if let Some(dict) = dict_obj.as_dict() {
                                                        let n = dict
                                                            .get("N")
                                                            .and_then(|o| o.as_integer())
                                                            .unwrap_or(3);
                                                        match n {
                                                            1 if !components.is_empty() => {
                                                                let g = components[0];
                                                                gs.stroke_color_rgb = (g, g, g);
                                                                handled = true;
                                                            },
                                                            3 if components.len() >= 3 => {
                                                                gs.stroke_color_rgb = (
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                );
                                                                handled = true;
                                                            },
                                                            4 if components.len() >= 4 => {
                                                                gs.stroke_color_rgb = cmyk_to_rgb(
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                    components[3],
                                                                );
                                                                handled = true;
                                                            },
                                                            _ => {},
                                                        }
                                                    }
                                                }
                                            },
                                            _ => {},
                                        }
                                    }
                                }
                            }
                            if !handled && !components.is_empty() {
                                let g = components[0];
                                gs.stroke_color_rgb = (g, g, g);
                            }
                        },
                    }
                    log::debug!(
                        "SetStrokeColor: {} {:?} -> {:?}",
                        space_name,
                        components,
                        gs.stroke_color_rgb
                    );
                },
                Operator::SetFillColorN { components, .. } => {
                    let gs = gs_stack.current_mut();
                    let space_name = gs.fill_color_space.clone();
                    let resolved_space = self.color_spaces.get(&space_name);

                    match space_name.as_str() {
                        "DeviceGray" | "G" if !components.is_empty() => {
                            let g = components[0];
                            gs.fill_color_rgb = (g, g, g);
                        },
                        "DeviceRGB" | "RGB" if components.len() >= 3 => {
                            gs.fill_color_rgb = (components[0], components[1], components[2]);
                        },
                        "DeviceCMYK" | "CMYK" if components.len() >= 4 => {
                            gs.fill_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        },
                        _ => {
                            let mut handled = false;
                            if let Some(rs) = resolved_space {
                                if let Some(arr) = rs.as_array() {
                                    if let Some(type_name) = arr.first().and_then(|o| o.as_name()) {
                                        match type_name {
                                            "ICCBased" if arr.len() > 1 => {
                                                if let Ok(dict_obj) = doc.resolve_object(&arr[1]) {
                                                    if let Some(dict) = dict_obj.as_dict() {
                                                        let n = dict
                                                            .get("N")
                                                            .and_then(|o| o.as_integer())
                                                            .unwrap_or(3);
                                                        match n {
                                                            1 if !components.is_empty() => {
                                                                let g = components[0];
                                                                gs.fill_color_rgb = (g, g, g);
                                                                handled = true;
                                                            },
                                                            3 if components.len() >= 3 => {
                                                                gs.fill_color_rgb = (
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                );
                                                                handled = true;
                                                            },
                                                            4 if components.len() >= 4 => {
                                                                gs.fill_color_rgb = cmyk_to_rgb(
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                    components[3],
                                                                );
                                                                handled = true;
                                                            },
                                                            _ => {},
                                                        }
                                                    }
                                                }
                                            },
                                            "Separation" | "DeviceN" => {
                                                if !components.is_empty() {
                                                    let g = 1.0 - components[0];
                                                    gs.fill_color_rgb = (g, g, g);
                                                    handled = true;
                                                }
                                            },
                                            _ => {},
                                        }
                                    }
                                }
                            }
                            if !handled && !components.is_empty() {
                                let g = components[0];
                                gs.fill_color_rgb = (g, g, g);
                            }
                        },
                    }
                    log::debug!(
                        "SetFillColorN: {} {:?} -> {:?}",
                        space_name,
                        components,
                        gs.fill_color_rgb
                    );
                },
                Operator::SetStrokeColorN { components, .. } => {
                    let gs = gs_stack.current_mut();
                    let space_name = gs.stroke_color_space.clone();
                    let resolved_space = self.color_spaces.get(&space_name);
                    match space_name.as_str() {
                        "DeviceGray" | "G" if !components.is_empty() => {
                            let g = components[0];
                            gs.stroke_color_rgb = (g, g, g);
                        },
                        "DeviceRGB" | "RGB" if components.len() >= 3 => {
                            gs.stroke_color_rgb = (components[0], components[1], components[2]);
                        },
                        "DeviceCMYK" | "CMYK" if components.len() >= 4 => {
                            gs.stroke_color_rgb = cmyk_to_rgb(
                                components[0],
                                components[1],
                                components[2],
                                components[3],
                            );
                        },
                        _ => {
                            let mut handled = false;
                            if let Some(rs) = resolved_space {
                                if let Some(arr) = rs.as_array() {
                                    if let Some(type_name) = arr.first().and_then(|o| o.as_name()) {
                                        match type_name {
                                            "ICCBased" if arr.len() > 1 => {
                                                if let Ok(dict_obj) = doc.resolve_object(&arr[1]) {
                                                    if let Some(dict) = dict_obj.as_dict() {
                                                        let n = dict
                                                            .get("N")
                                                            .and_then(|o| o.as_integer())
                                                            .unwrap_or(3);
                                                        match n {
                                                            1 if !components.is_empty() => {
                                                                let g = components[0];
                                                                gs.stroke_color_rgb = (g, g, g);
                                                                handled = true;
                                                            },
                                                            3 if components.len() >= 3 => {
                                                                gs.stroke_color_rgb = (
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                );
                                                                handled = true;
                                                            },
                                                            4 if components.len() >= 4 => {
                                                                gs.stroke_color_rgb = cmyk_to_rgb(
                                                                    components[0],
                                                                    components[1],
                                                                    components[2],
                                                                    components[3],
                                                                );
                                                                handled = true;
                                                            },
                                                            _ => {},
                                                        }
                                                    }
                                                }
                                            },
                                            _ => {},
                                        }
                                    }
                                }
                            }
                            if !handled && !components.is_empty() {
                                let g = components[0];
                                gs.stroke_color_rgb = (g, g, g);
                            }
                        },
                    }
                    log::debug!(
                        "SetStrokeColorN: {} {:?} -> {:?}",
                        space_name,
                        components,
                        gs.stroke_color_rgb
                    );
                },

                // Line style operators
                Operator::SetLineWidth { width } => {
                    gs_stack.current_mut().line_width = *width;
                },
                Operator::SetLineCap { cap_style } => {
                    gs_stack.current_mut().line_cap = *cap_style;
                },
                Operator::SetLineJoin { join_style } => {
                    gs_stack.current_mut().line_join = *join_style;
                },
                Operator::SetMiterLimit { limit } => {
                    gs_stack.current_mut().miter_limit = *limit;
                },
                Operator::SetDash { array, phase } => {
                    gs_stack.current_mut().dash_pattern = (array.clone(), *phase);
                },

                // Path construction
                Operator::MoveTo { x, y } => {
                    current_path.move_to(*x, *y);
                },
                Operator::LineTo { x, y } => {
                    current_path.line_to(*x, *y);
                },
                Operator::CurveTo {
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                } => {
                    current_path.cubic_to(*x1, *y1, *x2, *y2, *x3, *y3);
                },
                Operator::CurveToV { x2, y2, x3, y3 } => {
                    if let Some(last) = current_path.last_point() {
                        current_path.cubic_to(last.x, last.y, *x2, *y2, *x3, *y3);
                    }
                },
                Operator::CurveToY { x1, y1, x3, y3 } => {
                    current_path.cubic_to(*x1, *y1, *x3, *y3, *x3, *y3);
                },
                Operator::Rectangle {
                    x,
                    y,
                    width,
                    height,
                } => {
                    // Normalize negative width/height per PDF spec:
                    // re with negative dimensions means the rect extends in the opposite direction
                    let (nx, nw) = if *width < 0.0 {
                        (x + width, -width)
                    } else {
                        (*x, *width)
                    };
                    let (ny, nh) = if *height < 0.0 {
                        (y + height, -height)
                    } else {
                        (*y, *height)
                    };
                    if let Some(rect) = tiny_skia::Rect::from_xywh(nx, ny, nw, nh) {
                        current_path.push_rect(rect);
                    }
                },
                Operator::ClosePath => {
                    current_path.close();
                },

                // Path painting
                Operator::Stroke => {
                    apply_pending_clip(
                        &mut pending_clip,
                        &mut clip_stack,
                        pixmap,
                        base_transform,
                        &gs_stack,
                    );
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    if let Some(path) = current_path.finish() {
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        self.path_rasterizer
                            .stroke_path_clipped(pixmap, &path, transform, gs, clip);
                    }
                    current_path = PathBuilder::new();
                },
                Operator::Fill => {
                    apply_pending_clip(
                        &mut pending_clip,
                        &mut clip_stack,
                        pixmap,
                        base_transform,
                        &gs_stack,
                    );
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    if let Some(path) = current_path.finish() {
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        self.path_rasterizer.fill_path_clipped(
                            pixmap,
                            &path,
                            transform,
                            gs,
                            tiny_skia::FillRule::Winding,
                            clip,
                        );
                    }
                    current_path = PathBuilder::new();
                },
                Operator::FillStroke
                | Operator::CloseFillStroke
                | Operator::CloseFillStrokeEvenOdd => {
                    apply_pending_clip(
                        &mut pending_clip,
                        &mut clip_stack,
                        pixmap,
                        base_transform,
                        &gs_stack,
                    );
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    if let Some(path) = current_path.finish() {
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        // Fill first, then stroke on top
                        let fill_rule = if matches!(op, Operator::CloseFillStrokeEvenOdd) {
                            tiny_skia::FillRule::EvenOdd
                        } else {
                            tiny_skia::FillRule::Winding
                        };
                        self.path_rasterizer
                            .fill_path_clipped(pixmap, &path, transform, gs, fill_rule, clip);
                        self.path_rasterizer
                            .stroke_path_clipped(pixmap, &path, transform, gs, clip);
                    }
                    current_path = PathBuilder::new();
                },
                Operator::FillEvenOdd | Operator::FillStrokeEvenOdd => {
                    apply_pending_clip(
                        &mut pending_clip,
                        &mut clip_stack,
                        pixmap,
                        base_transform,
                        &gs_stack,
                    );
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    if let Some(path) = current_path.finish() {
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        self.path_rasterizer.fill_path_clipped(
                            pixmap,
                            &path,
                            transform,
                            gs,
                            tiny_skia::FillRule::EvenOdd,
                            clip,
                        );
                        // Add stroke for B* operator
                        if matches!(op, Operator::FillStrokeEvenOdd) {
                            self.path_rasterizer
                                .stroke_path_clipped(pixmap, &path, transform, gs, clip);
                        }
                    }
                    current_path = PathBuilder::new();
                },

                // Clipping
                Operator::ClipNonZero => {
                    if let Some(path) = current_path.clone().finish() {
                        pending_clip = Some((path, tiny_skia::FillRule::Winding));
                    }
                },
                Operator::ClipEvenOdd => {
                    if let Some(path) = current_path.clone().finish() {
                        pending_clip = Some((path, tiny_skia::FillRule::EvenOdd));
                    }
                },

                // Text object operators
                Operator::BeginText => {
                    in_text_object = true;
                    let gs = gs_stack.current_mut();
                    gs.text_matrix = Matrix::identity();
                    gs.text_line_matrix = Matrix::identity();
                    log::debug!("BT (BeginText)");
                },
                Operator::EndText => {
                    in_text_object = false;
                },

                // Text state operators
                Operator::Tc { char_space } => {
                    gs_stack.current_mut().char_space = *char_space;
                },
                Operator::Tw { word_space } => {
                    gs_stack.current_mut().word_space = *word_space;
                },
                Operator::Tz { scale } => {
                    gs_stack.current_mut().horizontal_scaling = *scale;
                },
                Operator::TL { leading } => {
                    gs_stack.current_mut().leading = *leading;
                },
                Operator::Ts { rise } => {
                    gs_stack.current_mut().text_rise = *rise;
                },
                Operator::Tr { render } => {
                    gs_stack.current_mut().render_mode = *render;
                },

                // Text showing
                Operator::Tj { text } => {
                    if in_text_object {
                        let clip = clip_stack.last().and_then(|c| c.as_ref());
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        let advance = self.text_rasterizer.render_text(
                            pixmap,
                            text,
                            transform,
                            gs,
                            resources,
                            doc,
                            clip,
                            &self.fonts,
                        )?;

                        // Advance text position: Tm = T(advance, 0) * Tm
                        let gs_mut = gs_stack.current_mut();
                        let advance_matrix = Matrix::translation(advance, 0.0);
                        gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                    }
                },
                Operator::Quote { text } => {
                    if in_text_object {
                        // Quote (') is T* followed by Tj
                        let gs_mut = gs_stack.current_mut();
                        let leading = gs_mut.leading;
                        let translation = Matrix::translation(0.0, -leading);
                        gs_mut.text_line_matrix = translation.multiply(&gs_mut.text_line_matrix);
                        gs_mut.text_matrix = gs_mut.text_line_matrix;

                        let clip = clip_stack.last().and_then(|c| c.as_ref());
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        log::debug!(
                            "' (Quote): rendering text at Tm=[{}, {}, {}, {}, {}, {}]",
                            gs.text_matrix.a,
                            gs.text_matrix.b,
                            gs.text_matrix.c,
                            gs.text_matrix.d,
                            gs.text_matrix.e,
                            gs.text_matrix.f
                        );
                        let advance = self.text_rasterizer.render_text(
                            pixmap,
                            text,
                            transform,
                            gs,
                            resources,
                            doc,
                            clip,
                            &self.fonts,
                        )?;

                        // Advance text position
                        let gs_mut = gs_stack.current_mut();
                        let advance_matrix = Matrix::translation(advance, 0.0);
                        gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                    }
                },
                Operator::TJ { array } => {
                    if in_text_object {
                        let clip = clip_stack.last().and_then(|c| c.as_ref());
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        log::debug!(
                            "TJ: rendering array at Tm=[{}, {}, {}, {}, {}, {}]",
                            gs.text_matrix.a,
                            gs.text_matrix.b,
                            gs.text_matrix.c,
                            gs.text_matrix.d,
                            gs.text_matrix.e,
                            gs.text_matrix.f
                        );
                        let advance = self.text_rasterizer.render_tj_array(
                            pixmap,
                            array,
                            transform,
                            gs,
                            resources,
                            doc,
                            clip,
                            &self.fonts,
                        )?;

                        // Advance text position: Tm = T(advance, 0) * Tm
                        let gs_mut = gs_stack.current_mut();
                        let advance_matrix = Matrix::translation(advance, 0.0);
                        gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                    }
                },
                Operator::DoubleQuote {
                    word_space,
                    char_space,
                    text,
                } => {
                    if in_text_object {
                        // Double Quote (") is Tw, Tc followed by ' (which is T*, Tj)
                        let gs_mut = gs_stack.current_mut();
                        gs_mut.word_space = *word_space;
                        gs_mut.char_space = *char_space;

                        let leading = gs_mut.leading;
                        let translation = Matrix::translation(0.0, -leading);
                        gs_mut.text_line_matrix = translation.multiply(&gs_mut.text_line_matrix);
                        gs_mut.text_matrix = gs_mut.text_line_matrix;

                        let clip = clip_stack.last().and_then(|c| c.as_ref());
                        let gs = gs_stack.current();
                        let transform = combine_transforms(base_transform, &gs.ctm);
                        log::debug!(
                            "\" (DoubleQuote): rendering text at Tm=[{}, {}, {}, {}, {}, {}]",
                            gs.text_matrix.a,
                            gs.text_matrix.b,
                            gs.text_matrix.c,
                            gs.text_matrix.d,
                            gs.text_matrix.e,
                            gs.text_matrix.f
                        );
                        let advance = self.text_rasterizer.render_text(
                            pixmap,
                            text,
                            transform,
                            gs,
                            resources,
                            doc,
                            clip,
                            &self.fonts,
                        )?;

                        // Advance text position
                        let gs_mut = gs_stack.current_mut();
                        let advance_matrix = Matrix::translation(advance, 0.0);
                        gs_mut.text_matrix = advance_matrix.multiply(&gs_mut.text_matrix);
                    }
                },

                // XObject (images)
                Operator::Do { name } => {
                    let gs = gs_stack.current();
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    log::debug!("Do: rendering XObject '{}'", name);
                    self.render_xobject(
                        pixmap, name, transform, gs, resources, doc, page_num, clip,
                    )?;
                },

                // Text positioning
                Operator::Td { tx, ty } => {
                    if in_text_object {
                        let gs = gs_stack.current_mut();
                        let translation = Matrix::translation(*tx, *ty);
                        gs.text_line_matrix = translation.multiply(&gs.text_line_matrix);
                        gs.text_matrix = gs.text_line_matrix;
                        log::debug!("Td: [{}, {}], text_matrix now: {:?}", tx, ty, gs.text_matrix);
                    }
                },
                Operator::TD { tx, ty } => {
                    if in_text_object {
                        let gs = gs_stack.current_mut();
                        gs.leading = -(*ty);
                        let translation = Matrix::translation(*tx, *ty);
                        gs.text_line_matrix = translation.multiply(&gs.text_line_matrix);
                        gs.text_matrix = gs.text_line_matrix;
                        log::debug!("TD: [{}, {}], text_matrix now: {:?}", tx, ty, gs.text_matrix);
                    }
                },
                Operator::Tm { a, b, c, d, e, f } => {
                    if in_text_object {
                        let gs = gs_stack.current_mut();
                        gs.text_matrix = Matrix {
                            a: *a,
                            b: *b,
                            c: *c,
                            d: *d,
                            e: *e,
                            f: *f,
                        };
                        gs.text_line_matrix = gs.text_matrix;
                        log::debug!(
                            "Tm: [{}, {}, {}, {}, {}, {}], text_matrix now: {:?}",
                            a,
                            b,
                            c,
                            d,
                            e,
                            f,
                            gs.text_matrix
                        );
                    }
                },
                Operator::TStar => {
                    if in_text_object {
                        let gs = gs_stack.current_mut();
                        let leading = gs.leading;
                        let translation = Matrix::translation(0.0, -leading);
                        gs.text_line_matrix = translation.multiply(&gs.text_line_matrix);
                        gs.text_matrix = gs.text_line_matrix;
                        log::debug!("T*: text_matrix now: {:?}", gs.text_matrix);
                    }
                },
                Operator::Tf { font, size } => {
                    let gs = gs_stack.current_mut();
                    gs.font_name = Some(font.clone());
                    gs.font_size = *size;
                },

                // Extended graphics state
                Operator::SetExtGState { dict_name } => {
                    self.apply_ext_g_state(gs_stack.current_mut(), dict_name, resources, doc)?;
                },

                // EndPath (n operator): discard current path without painting,
                // but apply any pending clip. Per PDF spec, W n is the standard
                // way to set a clipping path without filling or stroking.
                Operator::EndPath => {
                    apply_pending_clip(
                        &mut pending_clip,
                        &mut clip_stack,
                        pixmap,
                        base_transform,
                        &gs_stack,
                    );
                    current_path = PathBuilder::new();
                },

                // Shading (gradient) operator
                Operator::PaintShading { name } => {
                    let gs = gs_stack.current();
                    let transform = combine_transforms(base_transform, &gs.ctm);
                    let clip = clip_stack.last().and_then(|c| c.as_ref());
                    self.render_shading(pixmap, name, transform, gs, resources, doc, clip)?;
                },

                _ => {},
            }
        }

        Ok(())
    }

    /// Render a shading pattern (gradient).
    fn render_shading(
        &self,
        pixmap: &mut Pixmap,
        name: &str,
        transform: Transform,
        gs: &GraphicsState,
        resources: &Object,
        doc: &mut PdfDocument,
        clip_mask: Option<&tiny_skia::Mask>,
    ) -> Result<()> {
        // Look up shading resource
        let shading_dict = if let Object::Dictionary(res_dict) = resources {
            if let Some(shading_res) = res_dict.get("Shading") {
                let resolved = doc.resolve_object(shading_res)?;
                if let Some(shadings) = resolved.as_dict() {
                    if let Some(sh_obj) = shadings.get(name) {
                        let sh = doc.resolve_object(sh_obj)?;
                        sh.as_dict().cloned()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let shading = match shading_dict {
            Some(d) => d,
            None => {
                log::debug!("Shading '{}' not found in resources", name);
                return Ok(());
            },
        };

        let shading_type = shading
            .get("ShadingType")
            .and_then(|o| o.as_integer())
            .unwrap_or(0);

        match shading_type {
            2 => self.render_axial_shading(pixmap, &shading, transform, gs, doc, clip_mask),
            3 => self.render_radial_shading(pixmap, &shading, transform, gs, doc, clip_mask),
            _ => {
                log::debug!("Unsupported shading type {} for '{}'", shading_type, name);
                Ok(())
            },
        }
    }

    /// Render axial (linear) gradient shading (Type 2).
    fn render_axial_shading(
        &self,
        pixmap: &mut Pixmap,
        shading: &std::collections::HashMap<String, Object>,
        transform: Transform,
        gs: &GraphicsState,
        doc: &mut PdfDocument,
        clip_mask: Option<&tiny_skia::Mask>,
    ) -> Result<()> {
        // Parse Coords [x0 y0 x1 y1]
        let coords = shading.get("Coords").and_then(|o| o.as_array());
        let coords = match coords {
            Some(c) if c.len() >= 4 => c,
            _ => return Ok(()),
        };
        let get_f = |i: usize| -> f32 {
            match &coords[i] {
                Object::Real(v) => *v as f32,
                Object::Integer(v) => *v as f32,
                _ => 0.0,
            }
        };
        let (x0, y0, x1, y1) = (get_f(0), get_f(1), get_f(2), get_f(3));

        // Parse Extend [bool bool]
        let extend = shading.get("Extend").and_then(|o| o.as_array());
        let (extend_start, extend_end) = if let Some(ext) = extend {
            let e0 = ext
                .get(0)
                .map(|o| matches!(o, Object::Boolean(true)))
                .unwrap_or(false);
            let e1 = ext
                .get(1)
                .map(|o| matches!(o, Object::Boolean(true)))
                .unwrap_or(false);
            (e0, e1)
        } else {
            (false, false)
        };

        // Parse Function to get start and end colors
        // For simplicity, evaluate at t=0 and t=1 to get endpoint colors
        let (c0, c1) = self.evaluate_shading_function(shading, doc)?;

        // Transform gradient endpoints
        let mut p0 = tiny_skia::Point { x: x0, y: y0 };
        let mut p1 = tiny_skia::Point { x: x1, y: y1 };
        transform.map_point(&mut p0);
        transform.map_point(&mut p1);

        // Create gradient
        let spread = if extend_start && extend_end {
            tiny_skia::SpreadMode::Pad
        } else {
            tiny_skia::SpreadMode::Pad // tiny-skia default
        };

        let gradient = tiny_skia::LinearGradient::new(
            tiny_skia::Point { x: p0.x, y: p0.y },
            tiny_skia::Point { x: p1.x, y: p1.y },
            vec![
                tiny_skia::GradientStop::new(
                    0.0,
                    tiny_skia::Color::from_rgba(c0.0, c0.1, c0.2, gs.fill_alpha)
                        .unwrap_or(tiny_skia::Color::BLACK),
                ),
                tiny_skia::GradientStop::new(
                    1.0,
                    tiny_skia::Color::from_rgba(c1.0, c1.1, c1.2, gs.fill_alpha)
                        .unwrap_or(tiny_skia::Color::BLACK),
                ),
            ],
            spread,
            Transform::identity(),
        );

        if let Some(shader) = gradient {
            let mut paint = tiny_skia::Paint::default();
            paint.shader = shader;
            paint.anti_alias = true;

            // Fill entire pixmap with gradient (clipped by clip_mask)
            let rect =
                tiny_skia::Rect::from_xywh(0.0, 0.0, pixmap.width() as f32, pixmap.height() as f32)
                    .unwrap();
            let path = PathBuilder::from_rect(rect);
            pixmap.fill_path(
                &path,
                &paint,
                tiny_skia::FillRule::Winding,
                Transform::identity(),
                clip_mask,
            );
            log::debug!(
                "Rendered axial gradient from ({:.1},{:.1}) to ({:.1},{:.1})",
                p0.x,
                p0.y,
                p1.x,
                p1.y
            );
        }

        Ok(())
    }

    /// Render radial gradient shading (Type 3).
    fn render_radial_shading(
        &self,
        pixmap: &mut Pixmap,
        shading: &std::collections::HashMap<String, Object>,
        transform: Transform,
        gs: &GraphicsState,
        doc: &mut PdfDocument,
        clip_mask: Option<&tiny_skia::Mask>,
    ) -> Result<()> {
        // Parse Coords [x0 y0 r0 x1 y1 r1]
        let coords = shading.get("Coords").and_then(|o| o.as_array());
        let coords = match coords {
            Some(c) if c.len() >= 6 => c,
            _ => return Ok(()),
        };
        let get_f = |i: usize| -> f32 {
            match &coords[i] {
                Object::Real(v) => *v as f32,
                Object::Integer(v) => *v as f32,
                _ => 0.0,
            }
        };
        let (_x0, _y0, _r0, x1, y1, r1) =
            (get_f(0), get_f(1), get_f(2), get_f(3), get_f(4), get_f(5));

        let (c0, c1) = self.evaluate_shading_function(shading, doc)?;

        let mut center = tiny_skia::Point { x: x1, y: y1 };
        let mut edge = tiny_skia::Point { x: x1 + r1, y: y1 };
        transform.map_point(&mut center);
        transform.map_point(&mut edge);
        let radius = ((edge.x - center.x).powi(2) + (edge.y - center.y).powi(2)).sqrt();

        let gradient = tiny_skia::RadialGradient::new(
            tiny_skia::Point {
                x: center.x,
                y: center.y,
            },
            0.0, // start_radius (inner circle)
            tiny_skia::Point {
                x: center.x,
                y: center.y,
            },
            radius, // end_radius
            vec![
                tiny_skia::GradientStop::new(
                    0.0,
                    tiny_skia::Color::from_rgba(c0.0, c0.1, c0.2, gs.fill_alpha)
                        .unwrap_or(tiny_skia::Color::BLACK),
                ),
                tiny_skia::GradientStop::new(
                    1.0,
                    tiny_skia::Color::from_rgba(c1.0, c1.1, c1.2, gs.fill_alpha)
                        .unwrap_or(tiny_skia::Color::BLACK),
                ),
            ],
            tiny_skia::SpreadMode::Pad,
            Transform::identity(),
        );

        if let Some(shader) = gradient {
            let mut paint = tiny_skia::Paint::default();
            paint.shader = shader;
            paint.anti_alias = true;
            let rect =
                tiny_skia::Rect::from_xywh(0.0, 0.0, pixmap.width() as f32, pixmap.height() as f32)
                    .unwrap();
            let path = PathBuilder::from_rect(rect);
            pixmap.fill_path(
                &path,
                &paint,
                tiny_skia::FillRule::Winding,
                Transform::identity(),
                clip_mask,
            );
            log::debug!(
                "Rendered radial gradient at ({:.1},{:.1}) r={:.1}",
                center.x,
                center.y,
                radius
            );
        }

        Ok(())
    }

    /// Evaluate a shading function at t=0 and t=1 to get start/end colors.
    fn evaluate_shading_function(
        &self,
        shading: &std::collections::HashMap<String, Object>,
        doc: &mut PdfDocument,
    ) -> Result<((f32, f32, f32), (f32, f32, f32))> {
        // Try to parse a simple Type 2 (exponential interpolation) or Type 0 (sampled) function
        let func_obj = shading.get("Function");
        if let Some(func) = func_obj {
            let resolved = doc.resolve_object(func)?;
            if let Some(func_dict) = resolved.as_dict() {
                let func_type = func_dict
                    .get("FunctionType")
                    .and_then(|o| o.as_integer())
                    .unwrap_or(-1);

                if func_type == 2 {
                    // Type 2: Exponential interpolation f(x) = C0 + x^N * (C1 - C0)
                    let c0 = func_dict
                        .get("C0")
                        .and_then(|o| o.as_array())
                        .map(|arr| Self::parse_color_array(arr))
                        .unwrap_or((0.0, 0.0, 0.0));
                    let c1 = func_dict
                        .get("C1")
                        .and_then(|o| o.as_array())
                        .map(|arr| Self::parse_color_array(arr))
                        .unwrap_or((1.0, 1.0, 1.0));
                    return Ok((c0, c1));
                } else if func_type == 3 {
                    // Type 3: Stitching function — wraps multiple sub-functions
                    // For gradient endpoints, evaluate first sub-function at domain bounds
                    if let Some(funcs) = func_dict.get("Functions").and_then(|o| o.as_array()) {
                        if let Some(first_func) = funcs.first() {
                            let sub_resolved = doc.resolve_object(first_func)?;
                            if let Some(sub_dict) = sub_resolved.as_dict() {
                                let sub_type = sub_dict
                                    .get("FunctionType")
                                    .and_then(|o| o.as_integer())
                                    .unwrap_or(-1);
                                if sub_type == 2 {
                                    let c0 = sub_dict
                                        .get("C0")
                                        .and_then(|o| o.as_array())
                                        .map(|arr| Self::parse_color_array(arr))
                                        .unwrap_or((0.0, 0.0, 0.0));
                                    // For last color, check last sub-function if multiple
                                    let last_func_obj = funcs.last().unwrap_or(first_func);
                                    let last_resolved = doc.resolve_object(last_func_obj)?;
                                    let c1 = last_resolved
                                        .as_dict()
                                        .and_then(|d| d.get("C1"))
                                        .and_then(|o| o.as_array())
                                        .map(|arr| Self::parse_color_array(arr))
                                        .unwrap_or((1.0, 1.0, 1.0));
                                    return Ok((c0, c1));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    }

    fn parse_color_array(arr: &[Object]) -> (f32, f32, f32) {
        let get = |i: usize| -> f32 {
            arr.get(i)
                .map(|o| match o {
                    Object::Real(v) => *v as f32,
                    Object::Integer(v) => *v as f32,
                    _ => 0.0,
                })
                .unwrap_or(0.0)
        };
        if arr.len() >= 3 {
            (get(0), get(1), get(2))
        } else if arr.len() == 1 {
            let g = get(0);
            (g, g, g) // Grayscale
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Render an XObject (image or form).
    fn render_xobject(
        &mut self,
        pixmap: &mut Pixmap,
        name: &str,
        transform: Transform,
        gs: &GraphicsState,
        resources: &Object,
        doc: &mut PdfDocument,
        page_num: usize,
        clip_mask: Option<&tiny_skia::Mask>,
    ) -> Result<()> {
        // Get XObject from resources
        if let Object::Dictionary(res_dict) = resources {
            // PDF spec uses "XObject" (singular)
            if let Some(xobj_entry) = res_dict.get("XObject") {
                let xobjects_obj = doc.resolve_object(xobj_entry)?;
                if let Some(xobjects) = xobjects_obj.as_dict() {
                    if let Some(xobj_ref_obj) = xobjects.get(name) {
                        // Resolve reference if needed
                        let xobj = doc.resolve_object(xobj_ref_obj)?;
                        let xobj_ref = xobj_ref_obj.as_reference();
                        log::debug!("Resolved XObject '{}' type: {:?}", name, xobj);

                        if let Object::Stream { ref dict, .. } = xobj {
                            if let Some(smask) = dict.get("SMask") {
                                log::debug!("Image has SMask: {:?}", smask);
                            }
                            if let Some(mask) = dict.get("Mask") {
                                log::debug!("Image has Mask: {:?}", mask);
                            }
                            if let Some(imask) = dict.get("ImageMask") {
                                log::debug!("Image is ImageMask: {:?}", imask);
                            }
                            // Check subtype
                            if let Some(subtype) = dict.get("Subtype").and_then(|o| o.as_name()) {
                                match subtype {
                                    "Image" => {
                                        let smask = dict.get("SMask").cloned();
                                        let mask = dict.get("Mask").cloned();
                                        self.render_image(
                                            pixmap, &xobj, xobj_ref, transform, doc, clip_mask,
                                            smask, mask, gs,
                                        )?;
                                    },
                                    "Form" => {
                                        log::debug!("XObject '{}' is a Form", name);
                                        // Decoded stream data
                                        let stream_data = if let Some(r) = xobj_ref {
                                            doc.decode_stream_with_encryption(&xobj, r)?
                                        } else {
                                            xobj.decode_stream_data()?
                                        };

                                        // Form XObjects can have their own Resources dictionary.
                                        let form_resources =
                                            dict.get("Resources").unwrap_or(resources);

                                        // Save current fonts and load form-specific fonts
                                        let old_fonts = self.fonts.clone();
                                        let old_cs = self.color_spaces.clone();
                                        self.load_resources(doc, form_resources)?;

                                        self.render_form_xobject(
                                            pixmap,
                                            &dict,
                                            &stream_data,
                                            transform,
                                            doc,
                                            page_num,
                                            form_resources,
                                        )?;

                                        // Restore caches
                                        self.fonts = old_fonts;
                                        self.color_spaces = old_cs;
                                    },
                                    _ => {},
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Render an image XObject.
    fn render_image(
        &mut self,
        pixmap: &mut Pixmap,
        xobject: &Object,
        obj_ref: Option<ObjectRef>,
        transform: Transform,
        doc: &mut PdfDocument,
        clip_mask: Option<&tiny_skia::Mask>,
        smask_obj: Option<Object>,
        mask_obj: Option<Object>,
        gs: &GraphicsState,
    ) -> Result<()> {
        use crate::extractors::images::extract_image_from_xobject;

        // Use robust image extractor to handle various formats and color spaces
        let color_space_map = self.color_spaces.clone();
        let pdf_image =
            extract_image_from_xobject(Some(doc), xobject, obj_ref, Some(&color_space_map))?;
        let dynamic_image = pdf_image.to_dynamic_image()?;
        let mut rgba_image = dynamic_image.to_rgba8();

        // Handle /Mask (stencil mask image) — PDF spec section 8.9.6.2
        // The mask is a separate image whose samples define opacity (1=opaque, 0=transparent)
        if let Some(mask_ref) = mask_obj {
            if let Some(ref_obj) = mask_ref.as_reference() {
                if let Ok(mask_stream) = doc.load_object(ref_obj) {
                    // Try to decode the mask as an image
                    match extract_image_from_xobject(
                        Some(doc),
                        &mask_stream,
                        Some(ref_obj),
                        Some(&color_space_map),
                    ) {
                        Ok(mask_image) => {
                            if let Ok(mask_dyn) = mask_image.to_dynamic_image() {
                                let mask_gray = mask_dyn.to_luma8();
                                let mw = mask_gray.width();
                                let mh = mask_gray.height();
                                let iw = rgba_image.width();
                                let ih = rgba_image.height();
                                for y in 0..ih {
                                    for x in 0..iw {
                                        let mx = (x * mw / iw).min(mw - 1);
                                        let my = (y * mh / ih).min(mh - 1);
                                        let mask_val = mask_gray.get_pixel(mx, my)[0];
                                        let pixel = rgba_image.get_pixel_mut(x, y);
                                        pixel[3] =
                                            ((pixel[3] as u32 * mask_val as u32) / 255) as u8;
                                    }
                                }
                                log::debug!(
                                    "Applied image Mask ({}x{}) to image ({}x{})",
                                    mw,
                                    mh,
                                    iw,
                                    ih
                                );
                            }
                        },
                        Err(_) => {
                            // Fallback: decode stencil mask (ImageMask=true) directly from stream
                            if let Object::Stream { ref dict, .. } = mask_stream {
                                let mask_dict = dict;
                                let is_image_mask = mask_dict
                                    .get("ImageMask")
                                    .map(|o| matches!(o, Object::Boolean(true)))
                                    .unwrap_or(false);
                                if is_image_mask {
                                    let mw = mask_dict
                                        .get("Width")
                                        .and_then(|o| o.as_integer())
                                        .unwrap_or(0)
                                        as u32;
                                    let mh = mask_dict
                                        .get("Height")
                                        .and_then(|o| o.as_integer())
                                        .unwrap_or(0)
                                        as u32;
                                    if mw > 0 && mh > 0 {
                                        if let Ok(raw_mask_data) =
                                            doc.decode_stream_with_encryption(&mask_stream, ref_obj)
                                        {
                                            // CCITT data may be pass-through (not decompressed).
                                            // Check if we need to decompress Group 4 CCITT.
                                            let expected_bytes =
                                                ((mw as usize + 7) / 8) * mh as usize;
                                            let mask_data = if raw_mask_data.len()
                                                < expected_bytes / 2
                                            {
                                                // Data is still compressed — try Group 4 CCITT decompression
                                                let k = mask_dict
                                                    .get("DecodeParms")
                                                    .and_then(|o| o.as_dict())
                                                    .and_then(|d| d.get("K"))
                                                    .and_then(|o| o.as_integer())
                                                    .unwrap_or(0);
                                                if k == -1 {
                                                    #[allow(deprecated)]
                                                    let ccitt_result = crate::extractors::ccitt_bilevel::decompress_ccitt_group4(&raw_mask_data, mw, mh);
                                                    match ccitt_result {
                                                        Ok(decompressed) => {
                                                            log::debug!("CCITT Group4 decompressed mask: {} → {} bytes", raw_mask_data.len(), decompressed.len());
                                                            decompressed
                                                        },
                                                        Err(e) => {
                                                            log::debug!("CCITT decompression failed: {}, using raw data", e);
                                                            raw_mask_data
                                                        },
                                                    }
                                                } else {
                                                    raw_mask_data
                                                }
                                            } else {
                                                raw_mask_data
                                            };
                                            // 1-bit mask: each byte has 8 pixels, MSB first
                                            let iw = rgba_image.width();
                                            let ih = rgba_image.height();
                                            let row_bytes = (mw as usize + 7) / 8;
                                            for y in 0..ih {
                                                for x in 0..iw {
                                                    let mx = (x * mw / iw).min(mw - 1) as usize;
                                                    let my = (y * mh / ih).min(mh - 1) as usize;
                                                    let byte_idx = my * row_bytes + mx / 8;
                                                    let bit_idx = 7 - (mx % 8);
                                                    // PDF spec 8.9.6.2: mask bit 1 = paint (opaque), 0 = don't paint (transparent)
                                                    let mask_val = if byte_idx < mask_data.len() {
                                                        if (mask_data[byte_idx] >> bit_idx) & 1 == 1
                                                        {
                                                            255u8
                                                        } else {
                                                            0u8
                                                        }
                                                    } else {
                                                        255u8
                                                    };
                                                    let pixel = rgba_image.get_pixel_mut(x, y);
                                                    pixel[3] = ((pixel[3] as u32 * mask_val as u32)
                                                        / 255)
                                                        as u8;
                                                }
                                            }
                                            log::debug!("Applied stencil ImageMask ({}x{}) to image ({}x{})", mw, mh, iw, ih);
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
            // If Mask is an array, it's a color-key mask (not yet implemented)
        }

        // Handle SMask if present
        if let Some(smask_ref) = smask_obj {
            if let Ok(resolved_smask) = doc.resolve_object(&smask_ref) {
                let smask_obj_ref = smask_ref.as_reference();
                if let Ok(smask_image) = extract_image_from_xobject(
                    Some(doc),
                    &resolved_smask,
                    smask_obj_ref,
                    Some(&color_space_map),
                ) {
                    if let Ok(smask_dyn) = smask_image.to_dynamic_image() {
                        let smask_gray = smask_dyn.to_luma8();

                        // Apply SMask to alpha channel
                        // Rescale smask if dimensions don't match (simplification)
                        let sw = smask_gray.width();
                        let sh = smask_gray.height();
                        let iw = rgba_image.width();
                        let ih = rgba_image.height();

                        for y in 0..ih {
                            for x in 0..iw {
                                // Map image coordinate to smask coordinate
                                let sx = (x * sw / iw).min(sw - 1);
                                let sy = (y * sh / ih).min(sh - 1);
                                let alpha = smask_gray.get_pixel(sx, sy)[0];

                                let pixel = rgba_image.get_pixel_mut(x, y);
                                // Combine with existing alpha
                                pixel[3] = ((pixel[3] as u32 * alpha as u32) / 255) as u8;
                            }
                        }
                    }
                }
            }
        }

        let width = rgba_image.width();
        let height = rgba_image.height();

        // Create tiny-skia pixmap from RGBA data
        if let Some(img_pixmap) = Pixmap::from_vec(
            rgba_image.into_raw(),
            tiny_skia::IntSize::from_wh(width, height).unwrap(),
        ) {
            // PDF images are drawn in a unit square [0,1]x[0,1] in the current user space.
            // Image data is top-to-bottom, so we flip it to match PDF's bottom-to-top user space.
            let image_transform = transform
                .pre_translate(0.0, 1.0)
                .pre_scale(1.0 / width as f32, -1.0 / height as f32);

            // Draw image with transform and clip mask
            let mut paint = PixmapPaint::default();
            paint.opacity = gs.fill_alpha;
            paint.blend_mode = crate::rendering::pdf_blend_mode_to_skia(&gs.blend_mode);

            pixmap.draw_pixmap(0, 0, img_pixmap.as_ref(), &paint, image_transform, clip_mask);
        }

        Ok(())
    }

    /// Render a Form XObject by parsing its content stream recursively.
    ///
    /// Per PDF spec §8.10, a Form XObject contains its own content stream,
    /// optional /Matrix transform, and optional /Resources dictionary.
    fn render_form_xobject(
        &mut self,
        pixmap: &mut Pixmap,
        dict: &std::collections::HashMap<String, Object>,
        data: &[u8],
        parent_transform: Transform,
        doc: &mut PdfDocument,
        page_num: usize,
        parent_resources: &Object,
    ) -> Result<()> {
        // Parse /Matrix from form dict (default: identity)
        let form_matrix = if let Some(Object::Array(arr)) = dict.get("Matrix") {
            let get_f32 = |i: usize| -> f32 {
                match arr.get(i) {
                    Some(Object::Real(v)) => *v as f32,
                    Some(Object::Integer(v)) => *v as f32,
                    _ => {
                        if i == 0 || i == 3 {
                            1.0
                        } else {
                            0.0
                        }
                    },
                }
            };
            Transform::from_row(
                get_f32(0),
                get_f32(1),
                get_f32(2),
                get_f32(3),
                get_f32(4),
                get_f32(5),
            )
        } else {
            Transform::identity()
        };

        // Combine parent transform with form matrix
        let combined_transform = parent_transform.pre_concat(form_matrix);

        // Check for transparency group (PDF spec section 11.6.6)
        let is_transparency_group = dict
            .get("Group")
            .and_then(|g| g.as_dict())
            .map(|gd| gd.get("S").and_then(|s| s.as_name()) == Some("Transparency"))
            .unwrap_or(false);

        // Get form's /Resources (or fall back to parent resources)
        let form_resources = if let Some(res) = dict.get("Resources") {
            doc.resolve_object(res)?
        } else {
            parent_resources.clone()
        };

        // Parse form content stream
        let operators = match parse_content_stream(data) {
            Ok(ops) => ops,
            Err(e) => {
                return Err(e);
            },
        };

        if is_transparency_group {
            // Per PDF spec 11.6.6: Render transparency group to a separate pixmap,
            // then composite onto the parent. For isolated groups (I=true), the
            // initial backdrop is fully transparent.
            let is_isolated = dict
                .get("Group")
                .and_then(|g| g.as_dict())
                .and_then(|gd| gd.get("I"))
                .map(|i| match i {
                    Object::Boolean(b) => *b,
                    _ => false,
                })
                .unwrap_or(false);

            log::debug!("Rendering transparency group (isolated={})", is_isolated);

            // Create a separate pixmap for the group
            let mut group_pixmap =
                Pixmap::new(pixmap.width(), pixmap.height()).ok_or_else(|| {
                    crate::error::Error::InvalidPdf("Failed to create group pixmap".into())
                })?;

            if !is_isolated {
                // Non-isolated: copy parent content as initial backdrop
                group_pixmap.data_mut().copy_from_slice(pixmap.data());
            }
            // Isolated groups start fully transparent (default Pixmap state)

            // Execute operators into the group pixmap
            self.execute_operators(
                &mut group_pixmap,
                combined_transform,
                &operators,
                doc,
                page_num,
                &form_resources,
            )?;

            if is_isolated {
                // Composite the isolated group onto the parent using over blending
                pixmap.draw_pixmap(
                    0,
                    0,
                    group_pixmap.as_ref(),
                    &tiny_skia::PixmapPaint::default(),
                    Transform::identity(),
                    None,
                );
            } else {
                // Non-isolated: the group pixmap IS the result (it started with parent content)
                pixmap.data_mut().copy_from_slice(group_pixmap.data());
            }
        } else {
            // Non-group form XObject: render directly
            self.execute_operators(
                pixmap,
                combined_transform,
                &operators,
                doc,
                page_num,
                &form_resources,
            )?;
        }

        Ok(())
    }

    /// Apply extended graphics state parameters.
    fn apply_ext_g_state(
        &self,
        gs: &mut GraphicsState,
        dict_name: &str,
        resources: &Object,
        doc: &mut PdfDocument,
    ) -> Result<()> {
        if let Object::Dictionary(res_dict) = resources {
            if let Some(ext_gs_obj) = res_dict.get("ExtGState") {
                // Resolve ExtGState dictionary if it's a reference
                let ext_gs_resolved = doc.resolve_object(ext_gs_obj)?;
                if let Some(ext_g_states) = ext_gs_resolved.as_dict() {
                    if let Some(state_obj) = ext_g_states.get(dict_name) {
                        // Resolve individual state object (often a reference)
                        let state_resolved = doc.resolve_object(state_obj)?;
                        if let Some(state_dict) = state_resolved.as_dict() {
                            log::debug!(
                                "Applying ExtGState '{}': {:?}",
                                dict_name,
                                state_dict.keys()
                            );
                            // Apply transparency parameters
                            // PDF Spec Table 58: ca = non-stroking (fill) alpha
                            if let Some(ca) = state_dict.get("ca") {
                                let val = ca
                                    .as_real()
                                    .map(|v| v as f32)
                                    .or_else(|| ca.as_integer().map(|v| v as f32));
                                if let Some(v) = val {
                                    gs.fill_alpha = v;
                                    log::debug!("ExtGState: fill_alpha (ca) set to {}", v);
                                }
                            }

                            // PDF Spec Table 58: CA = stroking alpha
                            if let Some(ca_upper) = state_dict.get("CA") {
                                let val = ca_upper
                                    .as_real()
                                    .map(|v| v as f32)
                                    .or_else(|| ca_upper.as_integer().map(|v| v as f32));
                                if let Some(v) = val {
                                    gs.stroke_alpha = v;
                                    log::debug!("ExtGState: stroke_alpha (CA) set to {}", v);
                                }
                            }

                            if let Some(tk) = state_dict.get("TK") {
                                log::debug!("ExtGState: TK (Text Knockout) found: {:?}", tk);
                            }

                            if let Some(smask) = state_dict.get("SMask") {
                                log::debug!("ExtGState: SMask (Soft Mask) found: {:?}", smask);
                            }

                            if let Some(ais) = state_dict.get("AIS") {
                                log::debug!("ExtGState: AIS (Alpha Is Shape) found: {:?}", ais);
                            }

                            if let Some(bm) = state_dict.get("BM") {
                                let mode = match bm {
                                    Object::Name(n) => n.clone(),
                                    Object::Array(arr) => {
                                        // Sometimes BM is an array, use the first one
                                        arr.first()
                                            .and_then(|o| o.as_name())
                                            .unwrap_or("Normal")
                                            .to_string()
                                    },
                                    _ => "Normal".to_string(),
                                };
                                gs.blend_mode = mode;
                                log::debug!("ExtGState: blend_mode set to {}", gs.blend_mode);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Render annotations for a page.
    fn render_annotations(
        &mut self,
        pixmap: &mut Pixmap,
        base_transform: Transform,
        doc: &mut PdfDocument,
        page_num: usize,
    ) -> Result<()> {
        let annotations = doc.get_annotations(page_num)?;
        for annot in annotations {
            // Check if annotation has an appearance stream (/AP)
            if let Some(ap_obj) = annot.raw_dict.as_ref().and_then(|d| d.get("AP")) {
                let ap_stream_obj = doc.resolve_object(ap_obj)?;

                // Normal appearance (N)
                if let Object::Dictionary(ap_dict) = ap_stream_obj {
                    if let Some(n_entry) = ap_dict.get("N").or_else(|| ap_dict.values().next()) {
                        let n_stream_obj = doc.resolve_object(n_entry)?;
                        if let Object::Stream { ref dict, .. } = n_stream_obj {
                            let ap_data = if let Some(r) = n_entry.as_reference() {
                                doc.decode_stream_with_encryption(&n_stream_obj, r)?
                            } else {
                                n_stream_obj.decode_stream_data()?
                            };

                            if let Some(rect) = annot.rect {
                                let x = rect[0] as f32;
                                let y = rect[1] as f32;
                                let annot_transform = base_transform.pre_translate(x, y);

                                let old_fonts = self.fonts.clone();
                                let old_cs = self.color_spaces.clone();
                                if let Some(res) = dict.get("Resources") {
                                    if let Ok(res_obj) = doc.resolve_object(res) {
                                        self.load_resources(doc, &res_obj)?;
                                    }
                                }

                                self.render_form_xobject(
                                    pixmap,
                                    &dict,
                                    &ap_data,
                                    annot_transform,
                                    doc,
                                    page_num,
                                    &Object::Dictionary(std::collections::HashMap::new()),
                                )?;

                                self.fonts = old_fonts;
                                self.color_spaces = old_cs;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Encode Pixmap to JPEG format.
    fn encode_jpeg(&self, pixmap: &Pixmap) -> Result<Vec<u8>> {
        let width = pixmap.width();
        let height = pixmap.height();
        let data = pixmap.data();

        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
        for i in 0..(width * height) as usize {
            let r = data[i * 4] as f32;
            let g = data[i * 4 + 1] as f32;
            let b = data[i * 4 + 2] as f32;
            let a = data[i * 4 + 3] as f32 / 255.0;

            if a > 0.0 {
                rgb_data.push((r / a).min(255.0) as u8);
                rgb_data.push((g / a).min(255.0) as u8);
                rgb_data.push((b / a).min(255.0) as u8);
            } else {
                rgb_data.push(0);
                rgb_data.push(0);
                rgb_data.push(0);
            }
        }

        let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, rgb_data)
            .ok_or_else(|| Error::InvalidPdf("Failed to create image buffer".to_string()))?;

        let mut output = std::io::Cursor::new(Vec::new());
        img.write_to(&mut output, image::ImageFormat::Jpeg)
            .map_err(|e| Error::InvalidPdf(format!("JPEG encoding failed: {}", e)))?;

        Ok(output.into_inner())
    }
}

/// Combine two transformations.
fn combine_transforms(base: Transform, ctm: &Matrix) -> Transform {
    base.pre_concat(Transform::from_row(ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f))
}

/// Convert CMYK color components (0.0 to 1.0) to RGB (0.0 to 1.0).
fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32) {
    let r = (1.0 - c) * (1.0 - k);
    let g = (1.0 - m) * (1.0 - k);
    let b = (1.0 - y) * (1.0 - k);
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

fn apply_pending_clip(
    pending_clip: &mut Option<(tiny_skia::Path, tiny_skia::FillRule)>,
    clip_stack: &mut Vec<Option<tiny_skia::Mask>>,
    pixmap: &Pixmap,
    base_transform: Transform,
    gs_stack: &GraphicsStateStack,
) {
    if let Some((path, fill_rule)) = pending_clip.take() {
        let gs = gs_stack.current();
        let transform = combine_transforms(base_transform, &gs.ctm);

        if let Some(path_transformed) = path.transform(transform) {
            let bounds = path_transformed.bounds();
            log::debug!("Applying clip: fill_rule={:?}, bounds={:?}", fill_rule, bounds);

            let mut new_mask = tiny_skia::Mask::new(pixmap.width(), pixmap.height()).unwrap();
            new_mask.fill_path(
                &path_transformed,
                fill_rule,
                true, // anti-alias
                Transform::identity(),
            );

            if let Some(Some(current_mask)) = clip_stack.last() {
                // Intersect with existing mask
                let mut combined = current_mask.clone();
                let combined_data = combined.data_mut();
                let new_data = new_mask.data();
                for i in 0..combined_data.len() {
                    combined_data[i] = ((combined_data[i] as u32 * new_data[i] as u32) / 255) as u8;
                }
                *clip_stack.last_mut().unwrap() = Some(combined);
            } else {
                *clip_stack.last_mut().unwrap() = Some(new_mask);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Object;

    #[test]
    fn test_cmyk_to_rgb_white() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 0.0);
        assert!((r - 1.0).abs() < 0.001);
        assert!((g - 1.0).abs() < 0.001);
        assert!((b - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cmyk_to_rgb_black() {
        let (r, g, b) = cmyk_to_rgb(0.0, 0.0, 0.0, 1.0);
        assert!((r - 0.0).abs() < 0.001);
        assert!((g - 0.0).abs() < 0.001);
        assert!((b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cmyk_to_rgb_pure_cyan() {
        let (r, g, b) = cmyk_to_rgb(1.0, 0.0, 0.0, 0.0);
        assert!((r - 0.0).abs() < 0.001);
        assert!((g - 1.0).abs() < 0.001);
        assert!((b - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_color_array_rgb() {
        let arr = vec![Object::Real(0.5), Object::Real(0.25), Object::Real(0.75)];
        let (r, g, b) = PageRenderer::parse_color_array(&arr);
        assert!((r - 0.5).abs() < 0.001);
        assert!((g - 0.25).abs() < 0.001);
        assert!((b - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_parse_color_array_grayscale() {
        let arr = vec![Object::Real(0.5)];
        let (r, g, b) = PageRenderer::parse_color_array(&arr);
        assert!((r - 0.5).abs() < 0.001);
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn test_parse_color_array_integers() {
        let arr = vec![Object::Integer(1), Object::Integer(0), Object::Integer(0)];
        let (r, g, b) = PageRenderer::parse_color_array(&arr);
        assert!((r - 1.0).abs() < 0.001);
        assert!((g - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_negative_rect_normalization() {
        // Negative height: re 100 200 50 -30 → should normalize to (100, 170, 50, 30)
        let x: f32 = 100.0;
        let y: f32 = 200.0;
        let w: f32 = 50.0;
        let h: f32 = -30.0;
        let (nx, nw) = if w < 0.0 { (x + w, -w) } else { (x, w) };
        let (ny, nh) = if h < 0.0 { (y + h, -h) } else { (y, h) };
        assert!((nx - 100.0).abs() < 0.001);
        assert!((ny - 170.0).abs() < 0.001);
        assert!((nw - 50.0).abs() < 0.001);
        assert!((nh - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_negative_rect_both_negative() {
        let x: f32 = 100.0;
        let y: f32 = 200.0;
        let w: f32 = -50.0;
        let h: f32 = -30.0;
        let (nx, nw) = if w < 0.0 { (x + w, -w) } else { (x, w) };
        let (ny, nh) = if h < 0.0 { (y + h, -h) } else { (y, h) };
        assert!((nx - 50.0).abs() < 0.001);
        assert!((ny - 170.0).abs() < 0.001);
        assert!((nw - 50.0).abs() < 0.001);
        assert!((nh - 30.0).abs() < 0.001);
    }
}
