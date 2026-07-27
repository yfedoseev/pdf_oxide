/**
 * Manager for text search operations in PDF documents
 *
 * Caching is handled automatically at the Rust FFI layer, eliminating
 * the need for duplicate cache implementations in the binding.
 *
 * @example
 * ```typescript
 * import { SearchManager, SearchOptionsBuilder } from 'pdf_oxide';
 *
 * const doc = PdfDocument.open('document.pdf');
 * const searchManager = new SearchManager(doc);
 *
 * // Simple search
 * const results = searchManager.search('keyword');
 *
 * // Search with options
 * const options = SearchOptionsBuilder.strict().build();
 * const results = searchManager.search('keyword', options);
 *
 * // Count occurrences
 * const count = searchManager.countOccurrences('keyword');
 * ```
 */

export interface SearchResult {
  text?: string;
  pageIndex?: number;
  position?: number;
  boundingBox?: Record<string, number>;
  [key: string]: any;
}

export interface SearchStatistics {
  searchText: string;
  totalOccurrences: number;
  pagesContaining: number;
  firstMatchPage: number;
  lastMatchPage: number;
  pages: number[];
  occurrencesPerPage: Array<{
    pageIndex: number;
    pageNumber: number;
    count: number;
  }>;
}

export interface SearchCapabilities {
  caseSensitiveSearch: boolean;
  wholeWordSearch: boolean;
  regexSearch: boolean;
  annotationSearch: boolean;
  maxResults: number;
  isSearchable: boolean;
}

export class SearchManager {
  private _document: any;

  /**
   * Creates a new SearchManager for the given document
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
   * Searches for text in a specific page.
   * Results are automatically cached at the FFI layer.
   * @param searchText - Text to search for
   * @param pageIndex - Zero-based page index
   * @param options - Search options (caseSensitive, wholeWords, useRegex, etc.)
   * @returns Array of search results
   * @throws Error if parameters are invalid
   *
   * @example
   * ```typescript
   * const results = manager.search('error', 0);
   * results.forEach(result => {
   *   console.log(`Found at position ${result.position}`);
   * });
   * ```
   */
  search(searchText: string, pageIndex: number, options?: Record<string, any>): SearchResult[] {
    if (!searchText || typeof searchText !== 'string') {
      throw new Error('Search text must be a non-empty string');
    }

    if (typeof pageIndex !== 'number' || pageIndex < 0) {
      throw new Error('Page index must be a non-negative number');
    }

    if (pageIndex >= this._document.pageCount) {
      throw new Error(`Page index ${pageIndex} out of range`);
    }

    try {
      return this._document.search(searchText, pageIndex, options) || [];
    } catch (error) {
      throw new Error(`Search failed: ${(error as Error).message}`);
    }
  }

  /**
   * Searches for text across all pages
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns Array of search results with page information
   *
   * @example
   * ```typescript
   * const results = manager.searchAll('important');
   * console.log(`Found ${results.length} occurrences`);
   * ```
   */
  searchAll(searchText: string, options?: Record<string, any>): SearchResult[] {
    if (!searchText || typeof searchText !== 'string') {
      throw new Error('Search text must be a non-empty string');
    }

    const allResults: SearchResult[] = [];

    try {
      for (let i = 0; i < this._document.pageCount; i++) {
        const results = this.search(searchText, i, options);
        results.forEach((result) => {
          result.pageIndex = i;
          result.pageNumber = i + 1;
        });
        allResults.push(...results);
      }

      return allResults;
    } catch (error) {
      throw new Error(`Search all failed: ${(error as Error).message}`);
    }
  }

  /**
   * Counts occurrences of text in a page
   * @param searchText - Text to search for
   * @param pageIndex - Zero-based page index
   * @param options - Search options
   * @returns Number of occurrences found
   *
   * @example
   * ```typescript
   * const count = manager.countOccurrences('the', 0);
   * console.log(`"the" appears ${count} times on page 1`);
   * ```
   */
  countOccurrences(searchText: string, pageIndex: number, options?: Record<string, any>): number {
    const results = this.search(searchText, pageIndex, options);
    return results.length;
  }

  /**
   * Counts occurrences of text across all pages
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns Total occurrences
   *
   * @example
   * ```typescript
   * const totalCount = manager.countAllOccurrences('the');
   * console.log(`"the" appears ${totalCount} times in document`);
   * ```
   */
  countAllOccurrences(searchText: string, options?: Record<string, any>): number {
    const results = this.searchAll(searchText, options);
    return results.length;
  }

  /**
   * Builds the search index for every page up front, instead of the
   * lazy per-page build `search`/`searchAll` otherwise do on first use.
   *
   * @example
   * ```typescript
   * manager.prepareSearch();
   * for (const pattern of patterns) {
   *   const hits = manager.searchAll(pattern);
   * }
   * ```
   */
  prepareSearch(): void {
    this._document.prepareSearch();
  }

  /**
   * Drops the cached search index, if any, freeing its memory.
   * `search`/`searchAll` rebuild it lazily on next use.
   *
   * @example
   * ```typescript
   * manager.clearSearchIndex();
   * ```
   */
  clearSearchIndex(): void {
    this._document.clearSearchIndex();
  }

  /**
   * Checks if text exists in a page
   * @param searchText - Text to search for
   * @param pageIndex - Zero-based page index
   * @param options - Search options
   * @returns True if text found
   *
   * @example
   * ```typescript
   * if (manager.contains('error', 0)) {
   *   console.log('Page contains "error"');
   * }
   * ```
   */
  contains(searchText: string, pageIndex: number, options?: Record<string, any>): boolean {
    const results = this.search(searchText, pageIndex, options);
    return results.length > 0;
  }

  /**
   * Checks if text exists anywhere in document
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns True if text found anywhere
   *
   * @example
   * ```typescript
   * if (manager.containsAnywhere('copyright')) {
   *   console.log('Document contains copyright notice');
   * }
   * ```
   */
  containsAnywhere(searchText: string, options?: Record<string, any>): boolean {
    const results = this.searchAll(searchText, options);
    return results.length > 0;
  }

  /**
   * Gets pages containing the search text
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns Array of page indices (zero-based) containing the text
   *
   * @example
   * ```typescript
   * const pages = manager.getPagesContaining('error');
   * console.log(`"error" found on pages: ${pages.map(p => p + 1).join(', ')}`);
   * ```
   */
  getPagesContaining(searchText: string, options?: Record<string, any>): number[] {
    const results = this.searchAll(searchText, options);
    const pageSet = new Set(results.map((r) => r.pageIndex || 0));
    return Array.from(pageSet).sort((a, b) => a - b);
  }

  /**
   * Gets statistics for search results
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns Search statistics
   *
   * @example
   * ```typescript
   * const stats = manager.getSearchStatistics('error');
   * console.log(`Found ${stats.totalOccurrences} occurrences`);
   * console.log(`On ${stats.pagesContaining} pages`);
   * console.log(`First match on page ${stats.firstMatchPage + 1}`);
   * ```
   */
  getSearchStatistics(searchText: string, options?: Record<string, any>): SearchStatistics {
    const results = this.searchAll(searchText, options);

    // Extract unique pages and calculate per-page counts in single pass
    const pageMap = new Map<number, number>();
    for (const result of results) {
      const pageIdx = result.pageIndex || 0;
      if (!pageMap.has(pageIdx)) {
        pageMap.set(pageIdx, 0);
      }
      pageMap.set(pageIdx, (pageMap.get(pageIdx) || 0) + 1);
    }

    const pages = Array.from(pageMap.keys()).sort((a, b) => a - b);

    return {
      searchText,
      totalOccurrences: results.length,
      pagesContaining: pages.length,
      firstMatchPage: pages.length > 0 ? (pages[0] as number) : -1,
      lastMatchPage: pages.length > 0 ? (pages[pages.length - 1] as number) : -1,
      pages,
      occurrencesPerPage: pages.map((p) => ({
        pageIndex: p,
        pageNumber: p + 1,
        count: pageMap.get(p) || 0,
      })),
    };
  }

  /**
   * Searches with a regular expression
   * @param pattern - Regular expression pattern
   * @param options - Search options (will set useRegex: true)
   * @returns Array of search results
   *
   * @example
   * ```typescript
   * const results = manager.searchRegex(/error\d+/i);
   * // Finds "error1", "ERROR2", "Error3", etc.
   * ```
   */
  searchRegex(pattern: RegExp | string, options: Record<string, any> = {}): SearchResult[] {
    const regexStr = pattern instanceof RegExp ? pattern.source : pattern;

    if (!regexStr || typeof regexStr !== 'string') {
      throw new Error('Pattern must be a valid regular expression');
    }

    // Merge options and ensure useRegex is true
    const searchOptions = {
      ...options,
      useRegex: true,
    };

    try {
      return this.searchAll(regexStr, searchOptions);
    } catch (error) {
      throw new Error(`Regex search failed: ${(error as Error).message}`);
    }
  }

  /**
   * Finds first occurrence of text
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns First search result or null if not found
   *
   * @example
   * ```typescript
   * const first = manager.findFirst('chapter');
   * if (first) {
   *   console.log(`First "chapter" found on page ${first.pageNumber}`);
   * }
   * ```
   */
  findFirst(searchText: string, options?: Record<string, any>): SearchResult | null {
    const results = this.searchAll(searchText, options);
    return results.length > 0 ? (results[0] as SearchResult) : null;
  }

  /**
   * Finds last occurrence of text
   * @param searchText - Text to search for
   * @param options - Search options
   * @returns Last search result or null if not found
   */
  findLast(searchText: string, options?: Record<string, any>): SearchResult | null {
    const results = this.searchAll(searchText, options);
    return results.length > 0 ? (results[results.length - 1] as SearchResult) : null;
  }

  /**
   * Replaces text occurrences with highlighted versions (view only)
   * Gets all occurrences for highlighting without modification
   * @param searchText - Text to find
   * @param options - Search options
   * @returns Results formatted for highlighting
   *
   * @example
   * ```typescript
   * const highlights = manager.highlightMatches('important');
   * // Use results for UI highlighting
   * ```
   */
  highlightMatches(searchText: string, options?: Record<string, any>): SearchResult[] {
    return this.searchAll(searchText, options);
  }

  /**
   * Checks if document is searchable
   * @returns True if document supports text search
   */
  isSearchable(): boolean {
    try {
      // Try searching for common text to verify searchability
      this.searchAll('test');
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Gets search capabilities summary
   * @returns Search capabilities information
   */
  getCapabilities(): SearchCapabilities {
    return {
      caseSensitiveSearch: true,
      wholeWordSearch: true,
      regexSearch: true,
      annotationSearch: true,
      maxResults: 1000,
      isSearchable: this.isSearchable(),
    };
  }
}
