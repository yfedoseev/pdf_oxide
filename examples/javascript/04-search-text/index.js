// Search for a term across all pages of a PDF and print matches.
// Run: node index.js document.pdf "query"

import { PdfDocument } from "pdf-oxide";

const path = process.argv[2];
const query = process.argv[3];
if (!path || !query) {
  console.error('Usage: node index.js <file.pdf> "query"');
  process.exit(1);
}

const doc = PdfDocument.open(path);
const pages = doc.pageCount();
console.log(`Searching for "${query}" in ${path} (${pages} pages)...\n`);

// Build the per-page search index for every page up front, instead of the
// lazy per-page build searchAll()/searchPage() otherwise do on first use.
// (This example scans extracted text directly below, but prepareSearch()
// still primes the cache those two native methods share.)
doc.prepareSearch();

const lowerQuery = query.toLowerCase();
let total = 0;
let pagesWithHits = 0;

for (let i = 0; i < pages; i++) {
  const text = doc.extractText(i);
  const lower = text.toLowerCase();
  let count = 0;
  let pos = 0;
  while ((pos = lower.indexOf(lowerQuery, pos)) !== -1) {
    count++;
    pos += lowerQuery.length;
  }
  if (count === 0) continue;

  pagesWithHits++;
  console.log(`Page ${i + 1}: ${count} match(es)`);
  const snippet = text.substring(0, 120).replace(/\n/g, " ");
  console.log(`  - "${snippet}..."`);
  total += count;
  console.log();
}

// Free the cached search index now that we're done searching — useful
// before heavy extraction work on the same document object.
doc.clearSearchIndex();

doc.close();
console.log(`Found ${total} total matches across ${pagesWithHits} pages.`);
process.exit(0);
