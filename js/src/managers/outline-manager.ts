/**
 * Manager for PDF document outlines (bookmarks)
 *
 * Provides functionality for reading and navigating the document's outline tree,
 * which represents the hierarchical bookmark structure.
 *
 * @example
 * ```typescript
 * import { OutlineManager } from 'pdf_oxide';
 *
 * const doc = PdfDocument.open('document.pdf');
 * const outlineManager = new OutlineManager(doc);
 *
 * if (outlineManager.hasOutlines()) {
 *   const outlines = outlineManager.getOutlines();
 *   console.log(`Found ${outlines.length} outline items`);
 * }
 * ```
 */

export interface OutlineItem {
  title: string;
  pageIndex: number;
  pageNumber: number | null;
  level: number;
}

export class OutlineManager {
  private _document: any;

  /**
   * Creates a new OutlineManager for the given document
   * @param document - The PDF document
   * @throws Error if document is null or undefined
   */
  constructor(document: any) {
    if (!document) {
      throw new Error('Document is required');
    }
    this._document = document;
  }

  /**
   * Checks if the document has an outline (bookmarks)
   * @returns True if the document has outlines
   */
  hasOutlines(): boolean {
    try {
      return this._document.hasOutlines();
    } catch (error) {
      return false;
    }
  }

  /**
   * Gets the number of top-level outline items
   * @returns Number of outline items
   */
  getOutlineCount(): number {
    try {
      return this._document.getOutlineCount();
    } catch (error) {
      return 0;
    }
  }

  /**
   * Gets all outline items (flattened)
   * @returns Array of outline items
   *
   * @example
   * ```typescript
   * const outlines = manager.getOutlines();
   * outlines.forEach(item => {
   *   console.log(`${item.title} -> Page ${item.pageNumber}`);
   * });
   * ```
   */
  getOutlines(): OutlineItem[] {
    try {
      const rawOutlines = this._document.getOutlines();
      // Convert native OutlineInfo to OutlineItem format expected by JS
      return rawOutlines.map((item: any) => ({
        title: item.title,
        pageIndex: item.pageIndex,
        pageNumber: item.pageIndex >= 0 ? item.pageIndex + 1 : null,
        level: item.level,
      }));
    } catch (error) {
      return [];
    }
  }

  /**
   * Finds an outline item by title (case-insensitive substring match)
   * @param titleFragment - Partial title to search for
   * @returns Matching outline item or null
   *
   * @example
   * ```typescript
   * const item = manager.findByTitle('Introduction');
   * if (item) {
   *   console.log(`Found: ${item.title} on page ${item.pageNumber}`);
   * }
   * ```
   */
  findByTitle(titleFragment: string): OutlineItem | null {
    if (!titleFragment || typeof titleFragment !== 'string') {
      throw new Error('Title fragment must be a non-empty string');
    }

    const outlines = this.getOutlines();
    const fragment = titleFragment.toLowerCase();

    for (const item of outlines) {
      if (item.title && item.title.toLowerCase().includes(fragment)) {
        return item;
      }
    }

    return null;
  }

  /**
   * Finds all outline items by title (case-insensitive substring match)
   * @param titleFragment - Partial title to search for
   * @returns Array of matching outline items
   */
  findAllByTitle(titleFragment: string): OutlineItem[] {
    if (!titleFragment || typeof titleFragment !== 'string') {
      throw new Error('Title fragment must be a non-empty string');
    }

    const outlines = this.getOutlines();
    const fragment = titleFragment.toLowerCase();

    return outlines.filter((item) => item.title && item.title.toLowerCase().includes(fragment));
  }

  /**
   * Gets outline items for a specific page
   * @param pageIndex - Zero-based page index
   * @returns Outline items on that page
   */
  getOutlinesForPage(pageIndex: number): OutlineItem[] {
    if (typeof pageIndex !== 'number' || pageIndex < 0) {
      throw new Error('Page index must be a non-negative number');
    }

    const outlines = this.getOutlines();
    return outlines.filter((item) => item.pageIndex === pageIndex);
  }

  /**
   * Checks if a specific page has outline items
   * @param pageIndex - Zero-based page index
   * @returns True if page has outline items
   */
  pageHasOutlines(pageIndex: number): boolean {
    if (typeof pageIndex !== 'number' || pageIndex < 0) {
      throw new Error('Page index must be a non-negative number');
    }

    return this.getOutlinesForPage(pageIndex).length > 0;
  }

  /**
   * Gets an outline item by index
   * @param index - Item index
   * @returns Outline item or null if not found
   */
  getOutlineAt(index: number): OutlineItem | null {
    if (typeof index !== 'number' || index < 0) {
      throw new Error('Index must be a non-negative number');
    }

    const outlines = this.getOutlines();
    return outlines[index] || null;
  }

  /**
   * Checks if a page number exists in the outline
   * @param pageNumber - One-based page number
   * @returns True if page appears in outline
   */
  containsPageNumber(pageNumber: number): boolean {
    if (typeof pageNumber !== 'number' || pageNumber < 1) {
      throw new Error('Page number must be a positive number');
    }

    const outlines = this.getOutlines();
    return outlines.some((item) => item.pageNumber === pageNumber);
  }

  /**
   * Plans (does not produce) a split of the document at outline /
   * bookmark boundaries (#482), mirroring the core
   * `plan_split_by_bookmarks`. Returns the planned segments.
   *
   * @param options - title-prefix filter / depth / front-matter controls
   * @returns Array of planned split segments
   */
  planSplitByBookmarks(options?: {
    titlePrefix?: string;
    ignoreCase?: boolean;
    level?: number;
    includeFrontMatter?: boolean;
  }): Array<{
    index: number;
    start_page: number;
    end_page: number;
    title: string | null;
    file_stem: string;
    page_label: string | null;
  }> {
    return this._document.planSplitByBookmarks(options ?? {});
  }
}
