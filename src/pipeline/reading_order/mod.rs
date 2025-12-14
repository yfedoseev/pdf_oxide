//! Reading order strategies for text extraction.
//!
//! This module provides pluggable strategies for determining the reading order
//! of text spans extracted from PDF pages.
//!
//! # Available Strategies
//!
//! - [`StructureTreeStrategy`]: Uses PDF structure tree MCIDs for Tagged PDFs
//! - [`GeometricStrategy`]: Column-aware geometric analysis
//! - [`XYCutStrategy`]: Recursive XY-Cut spatial partitioning (newspapers, academic papers)
//! - [`SimpleStrategy`]: Simple top-to-bottom, left-to-right ordering

mod geometric;
mod simple;
mod structure_tree;
mod xycut;

pub use geometric::GeometricStrategy;
pub use simple::SimpleStrategy;
pub use structure_tree::StructureTreeStrategy;
pub use xycut::XYCutStrategy;

use crate::error::Result;
use crate::geometry::Rect;
use crate::layout::TextSpan;
use crate::pipeline::config::ReadingOrderStrategyType;
use crate::pipeline::OrderedTextSpan;

/// Trait for determining reading order of text spans.
///
/// Implementations decide how to order spans for reading. This is a key
/// abstraction point as different PDF types require different strategies:
///
/// - Tagged PDFs: Use structure tree MCIDs (PDF-spec-compliant)
/// - Multi-column: Use geometric column detection
/// - Simple: Top-to-bottom, left-to-right
pub trait ReadingOrderStrategy: Send + Sync {
    /// Apply reading order to a collection of spans.
    ///
    /// # Arguments
    ///
    /// * `spans` - Unordered text spans extracted from the page
    /// * `context` - Optional context information (structure tree, page info)
    ///
    /// # Returns
    ///
    /// Spans with assigned reading order indices.
    fn apply(
        &self,
        spans: Vec<TextSpan>,
        context: &ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>>;

    /// Return the name of this strategy for debugging.
    fn name(&self) -> &'static str;
}

/// Context information for reading order determination.
#[derive(Debug, Default)]
pub struct ReadingOrderContext {
    /// Current page number (0-indexed).
    pub page_number: u32,

    /// Page bounding box (if available).
    pub page_bbox: Option<Rect>,

    /// Whether the document has a structure tree (Tagged PDF).
    pub has_structure_tree: bool,

    /// MCID to reading order mapping (if structure tree available).
    pub mcid_order: Option<Vec<u32>>,
}

impl ReadingOrderContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the page number.
    pub fn with_page(mut self, page_number: u32) -> Self {
        self.page_number = page_number;
        self
    }

    /// Set the page bounding box.
    pub fn with_bbox(mut self, bbox: Rect) -> Self {
        self.page_bbox = Some(bbox);
        self
    }

    /// Set MCID order from structure tree traversal.
    pub fn with_mcid_order(mut self, mcid_order: Vec<u32>) -> Self {
        self.has_structure_tree = true;
        self.mcid_order = Some(mcid_order);
        self
    }
}

/// Create a reading order strategy based on configuration.
pub fn create_strategy(
    config: &crate::pipeline::ReadingOrderConfig,
) -> Box<dyn ReadingOrderStrategy> {
    match config.strategy {
        ReadingOrderStrategyType::StructureTreeFirst => Box::new(StructureTreeStrategy::new()),
        ReadingOrderStrategyType::Geometric => Box::new(GeometricStrategy::new()),
        ReadingOrderStrategyType::XYCut => Box::new(XYCutStrategy::new()),
        ReadingOrderStrategyType::Simple => Box::new(SimpleStrategy),
    }
}
