#!/usr/bin/env ruby
# frozen_string_literal: true

# Phase 2 bindings repair tool.
#
# Reads:
#   - lib/pdf_oxide/ffi/bindings.rb  (the prepared bindings file)
#   - /tmp/real_symbols.txt          (nm dump of libpdf_oxide.so T-symbols)
#
# Writes:
#   - lib/pdf_oxide/ffi/bindings.rb  (rewritten in place)
#
# Algorithm:
#   * Scan the file line-by-line.  When we encounter an `attach_function :sym`
#     header, accumulate subsequent lines until square-bracket depth returns
#     to zero AND we've consumed the trailing return-type token.
#   * A block is "complete" when `[` count == `]` count and the consumed
#     text ends with `,\s*:<return_type>` followed by optional comment.
#   * Mark each detected block as kept / phantom / duplicate, then rewrite.
#   * Append skeleton declarations for cdylib symbols not yet declared.

require 'set'

ROOT = File.expand_path('..', __dir__)
BINDINGS_PATH = File.join(ROOT, 'lib/pdf_oxide/ffi/bindings.rb')
SYMBOLS_PATH  = ENV.fetch('REAL_SYMBOLS', '/tmp/real_symbols.txt')

real_symbols = File.read(SYMBOLS_PATH).split("\n").map(&:strip).reject(&:empty?).to_set
content = File.read(BINDINGS_PATH)
lines = content.lines

# Phase A: locate attach_function blocks via bracket balancing.
blocks = []
i = 0
while i < lines.length
  line = lines[i]
  if line =~ /^(\s*)attach_function\s+:([a-zA-Z0-9_]+)/
    indent = ::Regexp.last_match(1)
    sym = ::Regexp.last_match(2)
    start = i
    # Consume until brackets balance AND we've seen a trailing return-type.
    bracket_depth = 0
    seen_open_bracket = false
    j = i
    while j < lines.length
      # Strip comments after `#` (but only when not inside a string — for this
      # file there are no string literals containing `#`, so naive strip is fine).
      effective = lines[j].sub(/#.*$/, '')
      bracket_depth += effective.count('[')
      bracket_depth -= effective.count(']')
      seen_open_bracket ||= effective.include?('[')
      if seen_open_bracket && bracket_depth.zero?
        # Heuristic terminator: this line (or a following line) ends with
        # `, :<retval>` after the `]`.  If the closing `]` is on this line,
        # the return type might be on this line or the next.
        # Check current line:
        if effective =~ /\]\s*,\s*:[a-zA-Z_][a-zA-Z0-9_]*\s*$/
          break
        end
        # Check whether a following line begins with `:retval`.
        if j + 1 < lines.length && lines[j + 1] =~ /^\s*:[a-zA-Z_][a-zA-Z0-9_]*\s*$/
          j += 1
          break
        end
        # Otherwise keep scanning (rare — usually we break here).
        break if effective =~ /\][^\[\]]*$/ # closing bracket with no further `[`
      end
      j += 1
    end
    blocks << { start: start, finish: j, sym: sym, indent: indent }
    i = j + 1
  else
    i += 1
  end
end

# Phase B: classify each block.
seen = Set.new
phantom_count = 0
duplicate_count = 0
kept_count = 0
to_strip = {}

blocks.each do |b|
  sym = b[:sym]
  if !real_symbols.include?(sym)
    phantom_count += 1
    to_strip[b[:start]] = [b, :phantom]
  elsif seen.include?(sym)
    duplicate_count += 1
    to_strip[b[:start]] = [b, :duplicate]
  else
    seen << sym
    kept_count += 1
  end
end

# Phase C: emit rewritten lines.
out = []
i = 0
while i < lines.length
  if to_strip.key?(i)
    b, reason = to_strip[i]
    span = b[:finish] - b[:start] + 1
    label = reason == :phantom ? 'phantom (no upstream symbol)' : 'duplicate declaration'
    out << "#{b[:indent]}# REMOVED #{label}: :#{b[:sym]} (#{span} line#{'s' if span > 1})\n"
    (span - 1).times { out << "#{b[:indent]}#\n" }
    i = b[:finish] + 1
  else
    out << lines[i]
    i += 1
  end
end

# Phase D: append skeletons for missing cdylib symbols.
missing = (real_symbols - seen).to_a.sort
skeleton = +""
unless missing.empty?
  skeleton << "\n      # ============================================================\n"
  skeleton << "      # AUTO-REPAIR Phase 2: cdylib symbols not declared by the prepared\n"
  skeleton << "      # snapshot.  Generic signature so the gem loads; real wrappers must\n"
  skeleton << "      # be added by Phase 3 (extend) and Phase 4 (test/CI).\n"
  skeleton << "      # ============================================================\n\n"
  missing.each do |sym|
    skeleton << "      attach_function :#{sym}, [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :pointer, blocking: false\n"
  end
end

result_lines = out
tail_idx = nil
(result_lines.length - 1).downto(0) do |k|
  if result_lines[k] =~ /^\s*end\s*$/
    next_two = result_lines[k + 1, 2] || []
    if next_two.size == 2 && next_two.all? { |l| l =~ /^\s*end\s*$/ }
      tail_idx = k
      break
    end
  end
end

if tail_idx.nil?
  warn 'Could not locate Bindings module closure; appending at EOF.'
  result = result_lines.join + skeleton
else
  result = result_lines[0...tail_idx].join + skeleton + result_lines[tail_idx..].join
end

File.write(BINDINGS_PATH, result)

puts "Detected attach_function blocks: #{blocks.size}"
puts "Phantom blocks stripped:         #{phantom_count}"
puts "Duplicate blocks stripped:       #{duplicate_count}"
puts "Kept (real, first-seen):         #{kept_count}"
puts "Missing skeletons appended:      #{missing.size}"
puts "Total real symbols in cdylib:    #{real_symbols.size}"
puts "Total symbols now declared:      #{seen.size + missing.size}"
