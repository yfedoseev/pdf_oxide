//! Graphics state management for content stream execution.
//!
//! This module provides the graphics state machine that tracks transformations,
//! text positioning, colors, and other parameters as operators are executed.

use crate::geometry::Point;

/// A 2D transformation matrix.
///
/// PDF uses matrices of the form:
/// ```text
/// [ a  b  0 ]
/// [ c  d  0 ]
/// [ e  f  1 ]
/// ```
///
/// Where (a,b,c,d) define scaling/rotation/skewing and (e,f) define translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix {
    /// Horizontal scaling component
    pub a: f32,
    /// Rotation/skew component
    pub b: f32,
    /// Rotation/skew component
    pub c: f32,
    /// Vertical scaling component
    pub d: f32,
    /// Horizontal translation
    pub e: f32,
    /// Vertical translation
    pub f: f32,
}

impl Matrix {
    /// Create an identity matrix.
    ///
    /// The identity matrix represents no transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::Matrix;
    ///
    /// let m = Matrix::identity();
    /// assert_eq!(m.a, 1.0);
    /// assert_eq!(m.d, 1.0);
    /// assert_eq!(m.e, 0.0);
    /// assert_eq!(m.f, 0.0);
    /// ```
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Create a translation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::Matrix;
    ///
    /// let m = Matrix::translation(10.0, 20.0);
    /// assert_eq!(m.e, 10.0);
    /// assert_eq!(m.f, 20.0);
    /// ```
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            e: tx,
            f: ty,
        }
    }

    /// Create a scaling matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::Matrix;
    ///
    /// let m = Matrix::scaling(2.0, 3.0);
    /// assert_eq!(m.a, 2.0);
    /// assert_eq!(m.d, 3.0);
    /// ```
    pub fn scaling(sx: f32, sy: f32) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            e: 0.0,
            f: 0.0,
        }
    }

    /// Multiply this matrix with another matrix.
    ///
    /// Matrix multiplication is not commutative: A * B ≠ B * A.
    /// The result represents first applying `other`, then applying `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::Matrix;
    ///
    /// let m1 = Matrix::translation(10.0, 0.0);
    /// let m2 = Matrix::scaling(2.0, 2.0);
    /// let result = m1.multiply(&m2);
    /// ```
    pub fn multiply(&self, other: &Matrix) -> Matrix {
        Matrix {
            a: self.a * other.a + self.b * other.c,
            b: self.a * other.b + self.b * other.d,
            c: self.c * other.a + self.d * other.c,
            d: self.c * other.b + self.d * other.d,
            e: self.e * other.a + self.f * other.c + other.e,
            f: self.e * other.b + self.f * other.d + other.f,
        }
    }

    /// Transform a point using this matrix.
    ///
    /// Applies the transformation defined by this matrix to the given point.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::Matrix;
    /// use pdf_oxide::geometry::Point;
    ///
    /// let m = Matrix::translation(10.0, 20.0);
    /// let p = m.transform_point(5.0, 10.0);
    /// assert_eq!(p.x, 15.0);
    /// assert_eq!(p.y, 30.0);
    /// ```
    pub fn transform_point(&self, x: f32, y: f32) -> Point {
        Point {
            x: self.a * x + self.c * y + self.e,
            y: self.b * x + self.d * y + self.f,
        }
    }

    /// Get the determinant of this matrix.
    ///
    /// The determinant indicates if the matrix is invertible (non-zero)
    /// and the scaling factor it applies to areas.
    pub fn determinant(&self) -> f32 {
        self.a * self.d - self.b * self.c
    }

    /// Check if this matrix is invertible.
    ///
    /// A matrix is invertible if its determinant is non-zero.
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > f32::EPSILON
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Graphics state parameters.
///
/// Tracks all parameters that affect how content is rendered, including
/// transformations, colors, line styles, and text state.
#[derive(Debug, Clone)]
pub struct GraphicsState {
    /// Current transformation matrix (maps user space to device space)
    pub ctm: Matrix,
    /// Text matrix (maps text space to user space)
    pub text_matrix: Matrix,
    /// Text line matrix (saved position at start of line)
    pub text_line_matrix: Matrix,

    // Text state parameters
    /// Character spacing (Tc)
    pub char_space: f32,
    /// Word spacing (Tw)
    pub word_space: f32,
    /// Horizontal scaling percentage (Tz)
    pub horizontal_scaling: f32,
    /// Text leading (TL)
    pub leading: f32,
    /// Current font name
    pub font_name: Option<String>,
    /// Current font size (Tf)
    pub font_size: f32,
    /// Text rise (Ts)
    pub text_rise: f32,
    /// Text rendering mode (Tr)
    pub render_mode: u8,

    // Color parameters
    /// Fill color space name (DeviceRGB, DeviceCMYK, DeviceGray, etc.)
    pub fill_color_space: String,
    /// Stroke color space name (DeviceRGB, DeviceCMYK, DeviceGray, etc.)
    pub stroke_color_space: String,
    /// Fill color (RGB)
    pub fill_color_rgb: (f32, f32, f32),
    /// Stroke color (RGB)
    pub stroke_color_rgb: (f32, f32, f32),
    /// Fill color (CMYK) - optional, if CMYK color space is used
    pub fill_color_cmyk: Option<(f32, f32, f32, f32)>,
    /// Stroke color (CMYK) - optional, if CMYK color space is used
    pub stroke_color_cmyk: Option<(f32, f32, f32, f32)>,

    // Line parameters (for completeness, though mainly used for graphics)
    /// Line width
    pub line_width: f32,
    /// Line dash pattern ([on1, off1, on2, off2, ...], phase)
    /// Empty array means solid line
    pub dash_pattern: (Vec<f32>, f32),
    /// Line cap style (J): 0=butt cap, 1=round cap, 2=projecting square cap
    pub line_cap: u8,
    /// Line join style (j): 0=miter join, 1=round join, 2=bevel join
    pub line_join: u8,
    /// Miter limit (M): ratio of miter length to line width
    pub miter_limit: f32,
    /// Rendering intent (ri): color rendering intent for images and graphics
    pub rendering_intent: String,
    /// Flatness tolerance (i): precision for curve approximation (0-100)
    pub flatness: f32,

    // Transparency parameters (from ExtGState)
    /// Fill alpha/opacity (CA): 0.0 (transparent) to 1.0 (opaque)
    pub fill_alpha: f32,
    /// Stroke alpha/opacity (ca): 0.0 (transparent) to 1.0 (opaque)
    pub stroke_alpha: f32,
    /// Blend mode (BM): Normal, Multiply, Screen, Overlay, etc.
    pub blend_mode: String,
}

impl GraphicsState {
    /// Create a new graphics state with default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::GraphicsState;
    ///
    /// let state = GraphicsState::new();
    /// assert_eq!(state.font_size, 12.0);
    /// assert_eq!(state.horizontal_scaling, 100.0);
    /// ```
    pub fn new() -> Self {
        Self {
            ctm: Matrix::identity(),
            text_matrix: Matrix::identity(),
            text_line_matrix: Matrix::identity(),
            char_space: 0.0,
            word_space: 0.0,
            horizontal_scaling: 100.0,
            leading: 0.0,
            font_name: None,
            font_size: 12.0,
            text_rise: 0.0,
            render_mode: 0,
            fill_color_space: "DeviceGray".to_string(), // PDF default
            stroke_color_space: "DeviceGray".to_string(), // PDF default
            fill_color_rgb: (0.0, 0.0, 0.0),            // Black
            stroke_color_rgb: (0.0, 0.0, 0.0),          // Black
            fill_color_cmyk: None,                      // No CMYK color set initially
            stroke_color_cmyk: None,                    // No CMYK color set initially
            line_width: 1.0,
            dash_pattern: (Vec::new(), 0.0), // Solid line
            line_cap: 0,                     // Butt cap (PDF default)
            line_join: 0,                    // Miter join (PDF default)
            miter_limit: 10.0,               // PDF default miter limit
            rendering_intent: "RelativeColorimetric".to_string(), // PDF default
            flatness: 1.0,                   // PDF default flatness tolerance
            fill_alpha: 1.0,                 // Fully opaque (PDF default)
            stroke_alpha: 1.0,               // Fully opaque (PDF default)
            blend_mode: "Normal".to_string(), // Normal blend mode (PDF default)
        }
    }

    /// Check if the current line style is dashed (not solid).
    ///
    /// Returns true if the dash pattern is non-empty.
    pub fn is_dashed(&self) -> bool {
        !self.dash_pattern.0.is_empty()
    }

    /// Check if the current line style is a dotted line pattern.
    ///
    /// A dotted line typically has short equal on/off segments.
    pub fn is_dotted(&self) -> bool {
        if self.dash_pattern.0.len() >= 2 {
            let on = self.dash_pattern.0[0];
            let off = self.dash_pattern.0[1];
            // Consider it dotted if on/off are similar and small (< 5 units)
            on < 5.0 && off < 5.0 && (on - off).abs() < 2.0
        } else {
            false
        }
    }
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Stack of graphics states for save/restore operations.
///
/// PDF's q (save) and Q (restore) operators push and pop graphics states.
/// This allows temporary modifications to the graphics state that can be
/// easily reverted.
#[derive(Debug, Clone)]
pub struct GraphicsStateStack {
    stack: Vec<GraphicsState>,
}

impl GraphicsStateStack {
    /// Create a new graphics state stack with an initial state.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::GraphicsStateStack;
    ///
    /// let stack = GraphicsStateStack::new();
    /// assert_eq!(stack.depth(), 1);
    /// ```
    pub fn new() -> Self {
        Self {
            stack: vec![GraphicsState::new()],
        }
    }

    /// Get a reference to the current graphics state.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::GraphicsStateStack;
    ///
    /// let stack = GraphicsStateStack::new();
    /// let state = stack.current();
    /// assert_eq!(state.font_size, 12.0);
    /// ```
    pub fn current(&self) -> &GraphicsState {
        self.stack.last().expect("Stack should never be empty")
    }

    /// Get a mutable reference to the current graphics state.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::GraphicsStateStack;
    ///
    /// let mut stack = GraphicsStateStack::new();
    /// stack.current_mut().font_size = 14.0;
    /// assert_eq!(stack.current().font_size, 14.0);
    /// ```
    pub fn current_mut(&mut self) -> &mut GraphicsState {
        self.stack.last_mut().expect("Stack should never be empty")
    }

    /// Save the current graphics state (q operator).
    ///
    /// Pushes a copy of the current state onto the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::GraphicsStateStack;
    ///
    /// let mut stack = GraphicsStateStack::new();
    /// stack.save();
    /// assert_eq!(stack.depth(), 2);
    /// ```
    pub fn save(&mut self) {
        let state = self.current().clone();
        self.stack.push(state);
    }

    /// Restore the previous graphics state (Q operator).
    ///
    /// Pops the current state from the stack. If only one state remains
    /// (the initial state), this operation has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::content::GraphicsStateStack;
    ///
    /// let mut stack = GraphicsStateStack::new();
    /// stack.save();
    /// stack.save();
    /// stack.restore();
    /// assert_eq!(stack.depth(), 2);
    /// stack.restore();
    /// assert_eq!(stack.depth(), 1);
    /// stack.restore(); // No effect, can't pop last state
    /// assert_eq!(stack.depth(), 1);
    /// ```
    pub fn restore(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }

    /// Get the current stack depth.
    ///
    /// The depth is always at least 1 (the initial state).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

impl Default for GraphicsStateStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity();
        assert_eq!(m.a, 1.0);
        assert_eq!(m.b, 0.0);
        assert_eq!(m.c, 0.0);
        assert_eq!(m.d, 1.0);
        assert_eq!(m.e, 0.0);
        assert_eq!(m.f, 0.0);
    }

    #[test]
    fn test_matrix_translation() {
        let m = Matrix::translation(10.0, 20.0);
        assert_eq!(m.e, 10.0);
        assert_eq!(m.f, 20.0);

        let p = m.transform_point(5.0, 10.0);
        assert_eq!(p.x, 15.0);
        assert_eq!(p.y, 30.0);
    }

    #[test]
    fn test_matrix_scaling() {
        let m = Matrix::scaling(2.0, 3.0);
        assert_eq!(m.a, 2.0);
        assert_eq!(m.d, 3.0);

        let p = m.transform_point(10.0, 10.0);
        assert_eq!(p.x, 20.0);
        assert_eq!(p.y, 30.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let m1 = Matrix::translation(10.0, 20.0);
        let m2 = Matrix::scaling(2.0, 2.0);
        let result = m1.multiply(&m2);

        // m1.multiply(&m2) applies m2 first, then m1: first translate, then scale
        // So point (5,5) -> translate to (15,25) -> scale to (30,50)
        let p = result.transform_point(5.0, 5.0);
        assert_eq!(p.x, 30.0); // (5+10)*2
        assert_eq!(p.y, 50.0); // (5+20)*2
    }

    #[test]
    fn test_matrix_multiply_order() {
        let m1 = Matrix::translation(10.0, 0.0);
        let m2 = Matrix::scaling(2.0, 1.0);

        let r1 = m1.multiply(&m2);
        let r2 = m2.multiply(&m1);

        // Different results show multiplication is not commutative
        let p = Point { x: 5.0, y: 0.0 };
        let p1 = r1.transform_point(p.x, p.y);
        let p2 = r2.transform_point(p.x, p.y);

        assert_ne!(p1.x, p2.x);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::scaling(2.0, 3.0);
        assert_eq!(m.determinant(), 6.0);

        let m_identity = Matrix::identity();
        assert_eq!(m_identity.determinant(), 1.0);
    }

    #[test]
    fn test_matrix_invertible() {
        let m = Matrix::scaling(2.0, 3.0);
        assert!(m.is_invertible());

        let m_degenerate = Matrix {
            a: 1.0,
            b: 2.0,
            c: 2.0,
            d: 4.0,
            e: 0.0,
            f: 0.0,
        };
        assert!(!m_degenerate.is_invertible());
    }

    #[test]
    fn test_graphics_state_new() {
        let state = GraphicsState::new();
        assert_eq!(state.font_size, 12.0);
        assert_eq!(state.horizontal_scaling, 100.0);
        assert_eq!(state.char_space, 0.0);
        assert_eq!(state.word_space, 0.0);
        assert_eq!(state.leading, 0.0);
        assert!(state.font_name.is_none());
    }

    #[test]
    fn test_graphics_state_default() {
        let state = GraphicsState::default();
        assert_eq!(state.font_size, 12.0);
    }

    #[test]
    fn test_graphics_state_stack_new() {
        let stack = GraphicsStateStack::new();
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.current().font_size, 12.0);
    }

    #[test]
    fn test_graphics_state_stack_save_restore() {
        let mut stack = GraphicsStateStack::new();

        // Modify current state
        stack.current_mut().font_size = 14.0;
        assert_eq!(stack.current().font_size, 14.0);

        // Save state
        stack.save();
        assert_eq!(stack.depth(), 2);
        assert_eq!(stack.current().font_size, 14.0);

        // Modify again
        stack.current_mut().font_size = 16.0;
        assert_eq!(stack.current().font_size, 16.0);

        // Restore
        stack.restore();
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.current().font_size, 14.0);
    }

    #[test]
    fn test_graphics_state_stack_restore_limit() {
        let mut stack = GraphicsStateStack::new();
        assert_eq!(stack.depth(), 1);

        // Try to restore when only one state exists
        stack.restore();
        assert_eq!(stack.depth(), 1); // Should still have one state

        // Save and restore multiple times
        stack.save();
        stack.save();
        stack.save();
        assert_eq!(stack.depth(), 4);

        stack.restore();
        stack.restore();
        stack.restore();
        assert_eq!(stack.depth(), 1);

        // One more restore should have no effect
        stack.restore();
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_graphics_state_color() {
        let mut state = GraphicsState::new();
        assert_eq!(state.fill_color_rgb, (0.0, 0.0, 0.0));
        assert_eq!(state.stroke_color_rgb, (0.0, 0.0, 0.0));

        state.fill_color_rgb = (1.0, 0.0, 0.0);
        state.stroke_color_rgb = (0.0, 1.0, 0.0);

        assert_eq!(state.fill_color_rgb, (1.0, 0.0, 0.0));
        assert_eq!(state.stroke_color_rgb, (0.0, 1.0, 0.0));
    }

    #[test]
    fn test_graphics_state_clone() {
        let mut state1 = GraphicsState::new();
        state1.font_size = 14.0;

        let state2 = state1.clone();
        assert_eq!(state2.font_size, 14.0);
    }

    #[test]
    fn test_matrix_transform_origin() {
        let m = Matrix::identity();
        let p = m.transform_point(0.0, 0.0);
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_matrix_default() {
        let m = Matrix::default();
        assert_eq!(m.a, 1.0);
        assert_eq!(m.d, 1.0);
    }
}
