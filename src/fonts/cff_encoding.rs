//! CFF (Compact Font Format) encoding parser.
//!
//! Parses the built-in encoding table from CFF font programs to extract
//! character code → glyph name mappings. Per PDF spec §9.6.6.2, when no
//! /BaseEncoding is specified in the encoding dictionary, the implicit base
//! encoding shall be the font program's built-in encoding.

use std::collections::HashMap;

/// Standard CFF string IDs (SIDs) 0-390 per CFF specification Table A-2.
/// Only the subset needed for encoding (common glyph names).
fn sid_to_name(sid: u16) -> Option<&'static str> {
    // CFF predefined strings (SID 0-390)
    // Full table from Adobe CFF specification, Appendix A
    static SID_NAMES: &[&str] = &[
        ".notdef",             // 0
        "space",               // 1
        "exclam",              // 2
        "quotedbl",            // 3
        "numbersign",          // 4
        "dollar",              // 5
        "percent",             // 6
        "ampersand",           // 7
        "quoteright",          // 8
        "parenleft",           // 9
        "parenright",          // 10
        "asterisk",            // 11
        "plus",                // 12
        "comma",               // 13
        "hyphen",              // 14
        "period",              // 15
        "slash",               // 16
        "zero",                // 17
        "one",                 // 18
        "two",                 // 19
        "three",               // 20
        "four",                // 21
        "five",                // 22
        "six",                 // 23
        "seven",               // 24
        "eight",               // 25
        "nine",                // 26
        "colon",               // 27
        "semicolon",           // 28
        "less",                // 29
        "equal",               // 30
        "greater",             // 31
        "question",            // 32
        "at",                  // 33
        "A",                   // 34
        "B",                   // 35
        "C",                   // 36
        "D",                   // 37
        "E",                   // 38
        "F",                   // 39
        "G",                   // 40
        "H",                   // 41
        "I",                   // 42
        "J",                   // 43
        "K",                   // 44
        "L",                   // 45
        "M",                   // 46
        "N",                   // 47
        "O",                   // 48
        "P",                   // 49
        "Q",                   // 50
        "R",                   // 51
        "S",                   // 52
        "T",                   // 53
        "U",                   // 54
        "V",                   // 55
        "W",                   // 56
        "X",                   // 57
        "Y",                   // 58
        "Z",                   // 59
        "bracketleft",         // 60
        "backslash",           // 61
        "bracketright",        // 62
        "asciicircum",         // 63
        "underscore",          // 64
        "quoteleft",           // 65
        "a",                   // 66
        "b",                   // 67
        "c",                   // 68
        "d",                   // 69
        "e",                   // 70
        "f",                   // 71
        "g",                   // 72
        "h",                   // 73
        "i",                   // 74
        "j",                   // 75
        "k",                   // 76
        "l",                   // 77
        "m",                   // 78
        "n",                   // 79
        "o",                   // 80
        "p",                   // 81
        "q",                   // 82
        "r",                   // 83
        "s",                   // 84
        "t",                   // 85
        "u",                   // 86
        "v",                   // 87
        "w",                   // 88
        "x",                   // 89
        "y",                   // 90
        "z",                   // 91
        "braceleft",           // 92
        "bar",                 // 93
        "braceright",          // 94
        "asciitilde",          // 95
        "exclamdown",          // 96
        "cent",                // 97
        "sterling",            // 98
        "fraction",            // 99
        "yen",                 // 100
        "florin",              // 101
        "section",             // 102
        "currency",            // 103
        "quotesingle",         // 104
        "quotedblleft",        // 105
        "guillemotleft",       // 106
        "guilsinglleft",       // 107
        "guilsinglright",      // 108
        "fi",                  // 109
        "fl",                  // 110
        "endash",              // 111
        "dagger",              // 112
        "daggerdbl",           // 113
        "periodcentered",      // 114
        "paragraph",           // 115
        "bullet",              // 116
        "quotesinglbase",      // 117
        "quotedblbase",        // 118
        "quotedblright",       // 119
        "guillemotright",      // 120
        "ellipsis",            // 121
        "perthousand",         // 122
        "questiondown",        // 123
        "grave",               // 124
        "acute",               // 125
        "circumflex",          // 126
        "tilde",               // 127
        "macron",              // 128
        "breve",               // 129
        "dotaccent",           // 130
        "dieresis",            // 131
        "ring",                // 132
        "cedilla",             // 133
        "hungarumlaut",        // 134
        "ogonek",              // 135
        "caron",               // 136
        "emdash",              // 137
        "AE",                  // 138
        "ordfeminine",         // 139
        "Lslash",              // 140
        "Oslash",              // 141
        "OE",                  // 142
        "ordmasculine",        // 143
        "ae",                  // 144
        "dotlessi",            // 145
        "lslash",              // 146
        "oslash",              // 147
        "oe",                  // 148
        "germandbls",          // 149
        "onesuperior",         // 150
        "logicalnot",          // 151
        "mu",                  // 152
        "trademark",           // 153
        "Eth",                 // 154
        "onehalf",             // 155
        "plusminus",           // 156
        "Thorn",               // 157
        "onequarter",          // 158
        "divide",              // 159
        "brokenbar",           // 160
        "degree",              // 161
        "thorn",               // 162
        "threequarters",       // 163
        "twosuperior",         // 164
        "registered",          // 165
        "minus",               // 166
        "eth",                 // 167
        "multiply",            // 168
        "threesuperior",       // 169
        "copyright",           // 170
        "Aacute",              // 171
        "Acircumflex",         // 172
        "Adieresis",           // 173
        "Agrave",              // 174
        "Aring",               // 175
        "Atilde",              // 176
        "Ccedilla",            // 177
        "Eacute",              // 178
        "Ecircumflex",         // 179
        "Edieresis",           // 180
        "Egrave",              // 181
        "Iacute",              // 182
        "Icircumflex",         // 183
        "Idieresis",           // 184
        "Igrave",              // 185
        "Ntilde",              // 186
        "Oacute",              // 187
        "Ocircumflex",         // 188
        "Odieresis",           // 189
        "Ograve",              // 190
        "Otilde",              // 191
        "Scaron",              // 192
        "Uacute",              // 193
        "Ucircumflex",         // 194
        "Udieresis",           // 195
        "Ugrave",              // 196
        "Yacute",              // 197
        "Ydieresis",           // 198
        "Zcaron",              // 199
        "aacute",              // 200
        "acircumflex",         // 201
        "adieresis",           // 202
        "agrave",              // 203
        "aring",               // 204
        "atilde",              // 205
        "ccedilla",            // 206
        "eacute",              // 207
        "ecircumflex",         // 208
        "edieresis",           // 209
        "egrave",              // 210
        "iacute",              // 211
        "icircumflex",         // 212
        "idieresis",           // 213
        "igrave",              // 214
        "ntilde",              // 215
        "oacute",              // 216
        "ocircumflex",         // 217
        "odieresis",           // 218
        "ograve",              // 219
        "otilde",              // 220
        "scaron",              // 221
        "uacute",              // 222
        "ucircumflex",         // 223
        "udieresis",           // 224
        "ugrave",              // 225
        "yacute",              // 226
        "ydieresis",           // 227
        "zcaron",              // 228
        "exclamsmall",         // 229
        "Hungarumlautsmall",   // 230
        "dollaroldstyle",      // 231
        "dollarsuperior",      // 232
        "ampersandsmall",      // 233
        "Acutesmall",          // 234
        "parenleftsuperior",   // 235
        "parenrightsuperior",  // 236
        "twodotenleader",      // 237
        "onedotenleader",      // 238
        "zerooldstyle",        // 239
        "oneoldstyle",         // 240
        "twooldstyle",         // 241
        "threeoldstyle",       // 242
        "fouroldstyle",        // 243
        "fiveoldstyle",        // 244
        "sixoldstyle",         // 245
        "sevenoldstyle",       // 246
        "eightoldstyle",       // 247
        "nineoldstyle",        // 248
        "commasuperior",       // 249
        "threequartersemdash", // 250
        "periodsuperior",      // 251
        "questionsmall",       // 252
        "asuperior",           // 253
        "bsuperior",           // 254
        "centsuperior",        // 255
        "dsuperior",           // 256
        "esuperior",           // 257
        "isuperior",           // 258
        "lsuperior",           // 259
        "msuperior",           // 260
        "nsuperior",           // 261
        "osuperior",           // 262
        "rsuperior",           // 263
        "ssuperior",           // 264
        "tsuperior",           // 265
        "ff",                  // 266
        "ffi",                 // 267
        "ffl",                 // 268
        "parenleftinferior",   // 269
        "parenrightinferior",  // 270
        "Circumflexsmall",     // 271
        "hyphensuperior",      // 272
        "Gravesmall",          // 273
        "Asmall",              // 274
        "Bsmall",              // 275
        "Csmall",              // 276
        "Dsmall",              // 277
        "Esmall",              // 278
        "Fsmall",              // 279
        "Gsmall",              // 280
        "Hsmall",              // 281
        "Ismall",              // 282
        "Jsmall",              // 283
        "Ksmall",              // 284
        "Lsmall",              // 285
        "Msmall",              // 286
        "Nsmall",              // 287
        "Osmall",              // 288
        "Psmall",              // 289
        "Qsmall",              // 290
        "Rsmall",              // 291
        "Ssmall",              // 292
        "Tsmall",              // 293
        "Usmall",              // 294
        "Vsmall",              // 295
        "Wsmall",              // 296
        "Xsmall",              // 297
        "Ysmall",              // 298
        "Zsmall",              // 299
        "colonmonetary",       // 300
        "onefitted",           // 301
        "rupiah",              // 302
        "Tildesmall",          // 303
        "exclamdownsmall",     // 304
        "centoldstyle",        // 305
        "Lslashsmall",         // 306
        "Scaronsmall",         // 307
        "Zcaronsmall",         // 308
        "Dieresissmall",       // 309
        "Brevesmall",          // 310
        "Caronsmall",          // 311
        "Dotaccentsmall",      // 312
        "Macronsmall",         // 313
        "figuredash",          // 314
        "hypheninferior",      // 315
        "Ogoneksmall",         // 316
        "Ringsmall",           // 317
        "Cedillasmall",        // 318
        "questiondownsmall",   // 319
        "oneeighth",           // 320
        "threeeighths",        // 321
        "fiveeighths",         // 322
        "seveneighths",        // 323
        "onethird",            // 324
        "twothirds",           // 325
        "zerosuperior",        // 326
        "foursuperior",        // 327
        "fivesuperior",        // 328
        "sixsuperior",         // 329
        "sevensuperior",       // 330
        "eightsuperior",       // 331
        "ninesuperior",        // 332
        "zeroinferior",        // 333
        "oneinferior",         // 334
        "twoinferior",         // 335
        "threeinferior",       // 336
        "fourinferior",        // 337
        "fiveinferior",        // 338
        "sixinferior",         // 339
        "seveninferior",       // 340
        "eightinferior",       // 341
        "nineinferior",        // 342
        "centinferior",        // 343
        "dollarinferior",      // 344
        "periodinferior",      // 345
        "commainferior",       // 346
        "Agravesmall",         // 347
        "Aacutesmall",         // 348
        "Acircumflexsmall",    // 349
        "Atildesmall",         // 350
        "Adieresissmall",      // 351
        "Aringsmall",          // 352
        "AEsmall",             // 353
        "Ccedillasmall",       // 354
        "Egravesmall",         // 355
        "Eacutesmall",         // 356
        "Ecircumflexsmall",    // 357
        "Edieresissmall",      // 358
        "Igravesmall",         // 359
        "Iacutesmall",         // 360
        "Icircumflexsmall",    // 361
        "Idieresissmall",      // 362
        "Ethsmall",            // 363
        "Ntildesmall",         // 364
        "Ogravesmall",         // 365
        "Oacutesmall",         // 366
        "Ocircumflexsmall",    // 367
        "Otildesmall",         // 368
        "Odieresissmall",      // 369
        "OEsmall",             // 370
        "Oslashsmall",         // 371
        "Ugravesmall",         // 372
        "Uacutesmall",         // 373
        "Ucircumflexsmall",    // 374
        "Udieresissmall",      // 375
        "Yacutesmall",         // 376
        "Thornsmall",          // 377
        "Ydieresissmall",      // 378
        "001.000",             // 379
        "001.001",             // 380
        "001.002",             // 381
        "001.003",             // 382
        "Black",               // 383
        "Bold",                // 384
        "Book",                // 385
        "Light",               // 386
        "Medium",              // 387
        "Regular",             // 388
        "Roman",               // 389
        "Semibold",            // 390
    ];

    if (sid as usize) < SID_NAMES.len() {
        Some(SID_NAMES[sid as usize])
    } else {
        None
    }
}

/// Look up the SID for a standard glyph name by searching the predefined SID table.
fn glyph_name_to_sid(name: &str) -> Option<u16> {
    for sid in 0u16..391 {
        if sid_to_name(sid) == Some(name) {
            return Some(sid);
        }
    }
    None
}

/// Parse a CFF INDEX structure, returning byte slices for each entry.
fn parse_index(data: &[u8], offset: usize) -> Option<(Vec<&[u8]>, usize)> {
    if offset + 2 > data.len() {
        return None;
    }
    let count = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    if count == 0 {
        return Some((Vec::new(), offset + 2));
    }

    if offset + 3 > data.len() {
        return None;
    }
    let off_size = data[offset + 2] as usize;
    if off_size == 0 || off_size > 4 {
        return None;
    }

    let offset_array_start = offset + 3;
    let offset_array_len = (count + 1) * off_size;
    if offset_array_start + offset_array_len > data.len() {
        return None;
    }

    // Read offsets
    let mut offsets = Vec::with_capacity(count + 1);
    for i in 0..=count {
        let pos = offset_array_start + i * off_size;
        let mut val: u32 = 0;
        for j in 0..off_size {
            val = (val << 8) | data[pos + j] as u32;
        }
        offsets.push(val as usize);
    }

    let data_start = offset_array_start + offset_array_len;
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        let start = data_start + offsets[i] - 1; // CFF offsets are 1-based
        let end = data_start + offsets[i + 1] - 1;
        if start > data.len() || end > data.len() || start > end {
            return None;
        }
        entries.push(&data[start..end]);
    }

    let next_offset = data_start + offsets[count] - 1;
    Some((entries, next_offset))
}

/// Parse a CFF DICT operand (integer or real).
/// Returns (value, bytes consumed).
fn parse_dict_operand(data: &[u8], pos: usize) -> Option<(i32, usize)> {
    if pos >= data.len() {
        return None;
    }
    let b0 = data[pos] as i32;
    match b0 {
        // Integer: 1 byte
        32..=246 => Some((b0 - 139, 1)),
        // Integer: 2 bytes
        247..=250 => {
            if pos + 1 >= data.len() {
                return None;
            }
            let b1 = data[pos + 1] as i32;
            Some(((b0 - 247) * 256 + b1 + 108, 2))
        },
        251..=254 => {
            if pos + 1 >= data.len() {
                return None;
            }
            let b1 = data[pos + 1] as i32;
            Some((-(b0 - 251) * 256 - b1 - 108, 2))
        },
        // Integer: 3 bytes (16-bit)
        28 => {
            if pos + 2 >= data.len() {
                return None;
            }
            let val = i16::from_be_bytes([data[pos + 1], data[pos + 2]]) as i32;
            Some((val, 3))
        },
        // Integer: 5 bytes (32-bit)
        29 => {
            if pos + 4 >= data.len() {
                return None;
            }
            let val =
                i32::from_be_bytes([data[pos + 1], data[pos + 2], data[pos + 3], data[pos + 4]]);
            Some((val, 5))
        },
        // Real number (skip, we only need integers for encoding/charset offsets)
        30 => {
            let mut i = pos + 1;
            while i < data.len() {
                let nibble1 = (data[i] >> 4) & 0x0F;
                let nibble2 = data[i] & 0x0F;
                if nibble1 == 0x0F || nibble2 == 0x0F {
                    return Some((0, i - pos + 1));
                }
                i += 1;
            }
            None
        },
        _ => None,
    }
}

/// Parse a CFF Top DICT to extract encoding and charset offsets.
fn parse_top_dict(dict_data: &[u8]) -> (i32, i32) {
    let mut encoding_offset: i32 = 0; // Default: StandardEncoding
    let mut charset_offset: i32 = 0; // Default: ISOAdobe charset

    let mut pos = 0;
    let mut operand_stack: Vec<i32> = Vec::new();

    while pos < dict_data.len() {
        let b0 = dict_data[pos];
        if b0 <= 21 {
            // Operator
            let op = if b0 == 12 {
                pos += 1;
                if pos >= dict_data.len() {
                    break;
                }
                (12u16 << 8) | dict_data[pos] as u16
            } else {
                b0 as u16
            };

            match op {
                16 => {
                    // Encoding (operator 16)
                    if let Some(&val) = operand_stack.last() {
                        encoding_offset = val;
                    }
                },
                15 => {
                    // charset (operator 15)
                    if let Some(&val) = operand_stack.last() {
                        charset_offset = val;
                    }
                },
                _ => {},
            }

            operand_stack.clear();
            pos += 1;
        } else if let Some((val, consumed)) = parse_dict_operand(dict_data, pos) {
            operand_stack.push(val);
            pos += consumed;
        } else {
            pos += 1;
        }
    }

    (encoding_offset, charset_offset)
}

/// Parse the CFF charset table.
/// Returns GID → SID mapping (GID 0 is always .notdef).
fn parse_charset(data: &[u8], offset: usize, num_glyphs: usize) -> Option<Vec<u16>> {
    if offset >= data.len() {
        return None;
    }

    let mut sids = Vec::with_capacity(num_glyphs);
    sids.push(0); // GID 0 = .notdef (SID 0)

    let format = data[offset];
    let mut pos = offset + 1;

    match format {
        0 => {
            // Format 0: array of SIDs
            for _ in 1..num_glyphs {
                if pos + 1 >= data.len() {
                    break;
                }
                let sid = u16::from_be_bytes([data[pos], data[pos + 1]]);
                sids.push(sid);
                pos += 2;
            }
        },
        1 => {
            // Format 1: ranges with 1-byte count
            while sids.len() < num_glyphs && pos + 2 < data.len() {
                let first_sid = u16::from_be_bytes([data[pos], data[pos + 1]]);
                let n_left = data[pos + 2] as u16;
                pos += 3;
                for i in 0..=n_left {
                    if sids.len() >= num_glyphs {
                        break;
                    }
                    sids.push(first_sid + i);
                }
            }
        },
        2 => {
            // Format 2: ranges with 2-byte count
            while sids.len() < num_glyphs && pos + 3 < data.len() {
                let first_sid = u16::from_be_bytes([data[pos], data[pos + 1]]);
                let n_left = u16::from_be_bytes([data[pos + 2], data[pos + 3]]);
                pos += 4;
                for i in 0..=n_left {
                    if sids.len() >= num_glyphs {
                        break;
                    }
                    sids.push(first_sid + i);
                }
            }
        },
        _ => return None,
    }

    Some(sids)
}

/// Parse the CFF encoding table.
/// Returns character code → GID mapping.
fn parse_encoding_table(data: &[u8], offset: usize) -> Option<HashMap<u8, u16>> {
    if offset >= data.len() {
        return None;
    }

    let mut code_to_gid = HashMap::new();
    let format = data[offset] & 0x7F; // Bit 7 is supplement flag
    let has_supplement = (data[offset] & 0x80) != 0;
    let mut pos = offset + 1;

    match format {
        0 => {
            // Format 0: array of codes
            if pos >= data.len() {
                return None;
            }
            let n_codes = data[pos] as usize;
            pos += 1;
            for gid in 1..=n_codes {
                if pos >= data.len() {
                    break;
                }
                let code = data[pos];
                code_to_gid.insert(code, gid as u16);
                pos += 1;
            }
        },
        1 => {
            // Format 1: ranges
            if pos >= data.len() {
                return None;
            }
            let n_ranges = data[pos] as usize;
            pos += 1;
            let mut gid: u16 = 1;
            for _ in 0..n_ranges {
                if pos + 1 >= data.len() {
                    break;
                }
                let first = data[pos];
                let n_left = data[pos + 1] as u16;
                pos += 2;
                for i in 0..=n_left {
                    let code = first.wrapping_add(i as u8);
                    code_to_gid.insert(code, gid);
                    gid += 1;
                }
            }
        },
        _ => return None,
    }

    // Handle supplement (additional code → SID mappings)
    if has_supplement && pos < data.len() {
        let n_sups = data[pos] as usize;
        pos += 1;
        for _ in 0..n_sups {
            if pos + 2 >= data.len() {
                break;
            }
            let code = data[pos];
            let sid = u16::from_be_bytes([data[pos + 1], data[pos + 2]]);
            pos += 3;
            // For supplements, we use SID directly as a pseudo-GID
            // The caller will need to handle this via the charset
            code_to_gid.insert(code, sid);
        }
    }

    Some(code_to_gid)
}

/// Resolve a glyph name from a SID, using predefined strings and the
/// String INDEX from the CFF font.
fn resolve_glyph_name<'a>(sid: u16, string_index: &'a [&'a [u8]]) -> Option<String> {
    if sid <= 390 {
        sid_to_name(sid).map(|s| s.to_string())
    } else {
        // Custom string from String INDEX
        let idx = (sid - 391) as usize;
        if idx < string_index.len() {
            std::str::from_utf8(string_index[idx])
                .ok()
                .map(|s| s.to_string())
        } else {
            None
        }
    }
}

/// Extract the CFF table from an OpenType (sfnt) wrapper.
/// Returns the CFF data slice if found, or None if the data isn't an sfnt container.
fn extract_cff_from_opentype(data: &[u8]) -> Option<&[u8]> {
    if data.len() < 12 {
        return None;
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    // Check for OpenType "OTTO" or TrueType 0x00010000
    if magic != 0x4F54544F && magic != 0x00010000 {
        return None;
    }
    let num_tables = u16::from_be_bytes([data[4], data[5]]) as usize;
    // Table directory starts at offset 12
    let mut pos = 12;
    for _ in 0..num_tables {
        if pos + 16 > data.len() {
            return None;
        }
        let tag = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        let offset =
            u32::from_be_bytes([data[pos + 8], data[pos + 9], data[pos + 10], data[pos + 11]])
                as usize;
        let length = u32::from_be_bytes([
            data[pos + 12],
            data[pos + 13],
            data[pos + 14],
            data[pos + 15],
        ]) as usize;
        // CFF tag = 0x43464620 ("CFF ")
        if tag == 0x43464620 && offset + length <= data.len() {
            return Some(&data[offset..offset + length]);
        }
        pos += 16;
    }
    None
}

/// Extract the built-in encoding from a CFF font program.
///
/// Returns a HashMap mapping character codes (u8) to Unicode characters.
/// This implements the CFF encoding → charset → glyph name → Unicode pipeline.
/// Also handles OpenType-wrapped CFF data (FontFile3 with sfnt container).
pub fn parse_cff_encoding(font_data: &[u8]) -> Option<HashMap<u8, char>> {
    if font_data.len() < 4 {
        return None;
    }

    // If data is wrapped in an OpenType container, extract the CFF table
    let cff_data = if font_data[0] != 1 {
        if let Some(cff) = extract_cff_from_opentype(font_data) {
            log::debug!(
                "Extracted CFF table ({} bytes) from OpenType wrapper ({} bytes)",
                cff.len(),
                font_data.len()
            );
            cff
        } else {
            // Not CFF version 1 and not an OpenType wrapper
            log::debug!("CFF version {} not supported (expected 1)", font_data[0]);
            return None;
        }
    } else {
        font_data
    };

    if cff_data.len() < 4 || cff_data[0] != 1 {
        return None;
    }
    let hdr_size = cff_data[2] as usize;

    // Parse Name INDEX
    let (_, after_name) = parse_index(cff_data, hdr_size)?;

    // Parse Top DICT INDEX
    let (top_dicts, after_top_dict) = parse_index(cff_data, after_name)?;
    if top_dicts.is_empty() {
        return None;
    }

    // Parse String INDEX
    let (string_index, _after_string) = parse_index(cff_data, after_top_dict)?;

    // Parse Top DICT to get encoding and charset offsets
    let (encoding_offset, charset_offset) = parse_top_dict(top_dicts[0]);

    if encoding_offset == 1 {
        // ExpertEncoding — rarely used for text
        log::debug!("CFF uses ExpertEncoding (predefined)");
        return None;
    }

    if encoding_offset == 0 {
        // StandardEncoding — for fonts with custom charsets, build a GID-based
        // fallback map. This handles subset fonts where character codes equal
        // GIDs rather than standard encoding positions.
        if charset_offset > 2 {
            log::debug!("CFF StandardEncoding + custom charset; building charset-based map");
            let num_glyphs = 256usize;
            let charset_sids = parse_charset(cff_data, charset_offset as usize, num_glyphs)?;

            let mut encoding_map = HashMap::new();
            for (gid, &sid) in charset_sids.iter().enumerate() {
                if gid == 0 || gid > 255 {
                    continue;
                }
                if let Some(glyph_name) = resolve_glyph_name(sid, &string_index) {
                    if let Some(unicode_char) = super::font_dict::glyph_name_to_unicode(&glyph_name)
                    {
                        encoding_map.insert(gid as u8, unicode_char);
                    }
                }
            }
            if !encoding_map.is_empty() {
                log::debug!(
                    "CFF charset-based fallback: {} character mappings",
                    encoding_map.len()
                );
                return Some(encoding_map);
            }
        }
        log::debug!("CFF uses StandardEncoding (predefined)");
        return None;
    }

    // Custom encoding (encoding_offset > 1): parse it
    let code_to_gid = parse_encoding_table(cff_data, encoding_offset as usize)?;

    let max_gid = code_to_gid.values().max().copied().unwrap_or(0) as usize;
    let num_glyphs = max_gid + 10;

    // Parse charset (GID → SID mapping)
    let charset_sids = if charset_offset == 0 {
        (0..num_glyphs as u16).collect()
    } else if charset_offset == 1 || charset_offset == 2 {
        log::debug!("CFF uses predefined charset {}", charset_offset);
        return None;
    } else {
        parse_charset(cff_data, charset_offset as usize, num_glyphs)?
    };

    // Build the final encoding map: code → Unicode
    let mut encoding_map = HashMap::new();

    for (&code, &gid) in &code_to_gid {
        let sid = if (gid as usize) < charset_sids.len() {
            charset_sids[gid as usize]
        } else {
            continue;
        };

        if let Some(glyph_name) = resolve_glyph_name(sid, &string_index) {
            if let Some(unicode_char) = super::font_dict::glyph_name_to_unicode(&glyph_name) {
                encoding_map.insert(code, unicode_char);
            }
        }
    }

    if encoding_map.is_empty() {
        None
    } else {
        log::debug!("CFF built-in encoding parsed: {} character mappings", encoding_map.len());
        Some(encoding_map)
    }
}

/// Parse a CFF font program and return a byte_code → glyph_id mapping.
/// This allows rendering CFF subset fonts by mapping PDF character codes
/// directly to glyph indices without needing a Unicode cmap.
pub fn parse_cff_gid_mapping(font_data: &[u8]) -> Option<HashMap<u8, u16>> {
    if font_data.len() < 4 {
        return None;
    }

    let cff_data = if font_data[0] != 1 {
        extract_cff_from_opentype(font_data)?
    } else {
        font_data
    };

    if cff_data.len() < 4 || cff_data[0] != 1 {
        return None;
    }
    let hdr_size = cff_data[2] as usize;

    let (_, after_name) = parse_index(cff_data, hdr_size)?;
    let (top_dicts, after_top_dict) = parse_index(cff_data, after_name)?;
    if top_dicts.is_empty() {
        return None;
    }

    let (_string_index, _after_string) = parse_index(cff_data, after_top_dict)?;
    let (encoding_offset, charset_offset) = parse_top_dict(top_dicts[0]);

    if encoding_offset == 0 && charset_offset > 2 {
        // StandardEncoding + custom charset:
        // byte_code → SID (via CFF Standard Encoding) → GID (via charset)
        let num_glyphs = 256usize;
        if let Some(charset_sids) = parse_charset(cff_data, charset_offset as usize, num_glyphs) {
            // Build SID → GID reverse map from charset
            let mut sid_to_gid: HashMap<u16, u16> = HashMap::new();
            for (gid, &sid) in charset_sids.iter().enumerate() {
                if gid > 0 {
                    sid_to_gid.entry(sid).or_insert(gid as u16);
                }
            }
            // CFF Standard Encoding: byte_code → SID
            // Map byte codes through standard encoding to SIDs, then to GIDs
            let mut map = HashMap::new();
            for byte_code in 0u16..256 {
                // Get the glyph name for this byte code using standard encoding
                let glyph_name = super::font_dict::FontInfo::gid_to_standard_glyph_name(byte_code);
                if let Some(name) = glyph_name {
                    // Find the SID for this glyph name
                    if let Some(sid) = glyph_name_to_sid(name) {
                        if let Some(&gid) = sid_to_gid.get(&sid) {
                            map.insert(byte_code as u8, gid);
                        }
                    }
                }
            }
            if !map.is_empty() {
                log::debug!("CFF StandardEncoding→charset GID mapping: {} entries", map.len());
                return Some(map);
            }
        }
        return None;
    }

    if encoding_offset <= 1 {
        return None;
    }

    // Custom encoding: parse byte_code → GID mapping directly
    parse_encoding_table(cff_data, encoding_offset as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // sid_to_name tests
    // ==========================================

    #[test]
    fn test_sid_to_name_notdef() {
        assert_eq!(sid_to_name(0), Some(".notdef"));
    }

    #[test]
    fn test_sid_to_name_space() {
        assert_eq!(sid_to_name(1), Some("space"));
    }

    #[test]
    fn test_sid_to_name_letters() {
        assert_eq!(sid_to_name(34), Some("A"));
        assert_eq!(sid_to_name(59), Some("Z"));
        assert_eq!(sid_to_name(66), Some("a"));
        assert_eq!(sid_to_name(91), Some("z"));
    }

    #[test]
    fn test_sid_to_name_digits() {
        assert_eq!(sid_to_name(17), Some("zero"));
        assert_eq!(sid_to_name(26), Some("nine"));
    }

    #[test]
    fn test_sid_to_name_punctuation() {
        assert_eq!(sid_to_name(2), Some("exclam"));
        assert_eq!(sid_to_name(15), Some("period"));
        assert_eq!(sid_to_name(13), Some("comma"));
    }

    #[test]
    fn test_sid_to_name_ligatures() {
        assert_eq!(sid_to_name(109), Some("fi"));
        assert_eq!(sid_to_name(110), Some("fl"));
        assert_eq!(sid_to_name(266), Some("ff"));
        assert_eq!(sid_to_name(267), Some("ffi"));
        assert_eq!(sid_to_name(268), Some("ffl"));
    }

    #[test]
    fn test_sid_to_name_accented() {
        assert_eq!(sid_to_name(171), Some("Aacute"));
        assert_eq!(sid_to_name(200), Some("aacute"));
        assert_eq!(sid_to_name(227), Some("ydieresis"));
    }

    #[test]
    fn test_sid_to_name_last_entries() {
        assert_eq!(sid_to_name(388), Some("Regular"));
        assert_eq!(sid_to_name(389), Some("Roman"));
        assert_eq!(sid_to_name(390), Some("Semibold"));
    }

    #[test]
    fn test_sid_to_name_out_of_range() {
        assert_eq!(sid_to_name(391), None);
        assert_eq!(sid_to_name(500), None);
        assert_eq!(sid_to_name(u16::MAX), None);
    }

    // ==========================================
    // parse_index tests
    // ==========================================

    #[test]
    fn test_parse_index_too_short() {
        assert_eq!(parse_index(&[0x00], 0), None);
    }

    #[test]
    fn test_parse_index_empty() {
        // count = 0
        let data = [0x00, 0x00];
        let result = parse_index(&data, 0);
        assert!(result.is_some());
        let (entries, next) = result.unwrap();
        assert!(entries.is_empty());
        assert_eq!(next, 2);
    }

    #[test]
    fn test_parse_index_single_entry() {
        // count=1, off_size=1, offsets=[1, 4] (entry is 3 bytes)
        let data = vec![
            0x00, 0x01, // count = 1
            0x01, // off_size = 1
            0x01, // offset[0] = 1
            0x04, // offset[1] = 4
            b'A', b'B', b'C', // data: "ABC"
        ];
        let result = parse_index(&data, 0);
        assert!(result.is_some());
        let (entries, _next) = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], b"ABC");
    }

    #[test]
    fn test_parse_index_multiple_entries() {
        // count=2, off_size=1, offsets=[1, 3, 5]
        let data = vec![
            0x00, 0x02, // count = 2
            0x01, // off_size = 1
            0x01, // offset[0] = 1
            0x03, // offset[1] = 3
            0x05, // offset[2] = 5
            b'H', b'i', // entry 0: "Hi"
            b'O', b'K', // entry 1: "OK"
        ];
        let result = parse_index(&data, 0);
        assert!(result.is_some());
        let (entries, _next) = result.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], b"Hi");
        assert_eq!(entries[1], b"OK");
    }

    #[test]
    fn test_parse_index_invalid_off_size_zero() {
        let data = vec![0x00, 0x01, 0x00]; // off_size = 0
        assert_eq!(parse_index(&data, 0), None);
    }

    #[test]
    fn test_parse_index_invalid_off_size_too_large() {
        let data = vec![0x00, 0x01, 0x05]; // off_size = 5
        assert_eq!(parse_index(&data, 0), None);
    }

    #[test]
    fn test_parse_index_truncated_offset_array() {
        // count=1, off_size=1, but not enough data for offsets
        let data = vec![0x00, 0x01, 0x01, 0x01]; // missing second offset
        assert_eq!(parse_index(&data, 0), None);
    }

    #[test]
    fn test_parse_index_with_offset() {
        // Parse index starting at offset 3
        let data = vec![
            0xFF, 0xFF, 0xFF, // padding
            0x00, 0x01, // count = 1
            0x01, // off_size = 1
            0x01, // offset[0] = 1
            0x02, // offset[1] = 2
            b'X', // data: "X"
        ];
        let result = parse_index(&data, 3);
        assert!(result.is_some());
        let (entries, _) = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], b"X");
    }

    #[test]
    fn test_parse_index_off_size_2() {
        // count=1, off_size=2, offsets=[0x0001, 0x0003]
        let data = vec![
            0x00, 0x01, // count = 1
            0x02, // off_size = 2
            0x00, 0x01, // offset[0] = 1
            0x00, 0x03, // offset[1] = 3
            b'A', b'B', // data: "AB"
        ];
        let result = parse_index(&data, 0);
        assert!(result.is_some());
        let (entries, _) = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], b"AB");
    }

    #[test]
    fn test_parse_index_data_out_of_bounds() {
        // offsets reference data beyond the buffer
        let data = vec![
            0x00, 0x01, // count = 1
            0x01, // off_size = 1
            0x01, // offset[0] = 1
            0xFF, // offset[1] = 255 (way out of bounds)
        ];
        assert_eq!(parse_index(&data, 0), None);
    }

    // ==========================================
    // parse_dict_operand tests
    // ==========================================

    #[test]
    fn test_parse_dict_operand_empty() {
        assert_eq!(parse_dict_operand(&[], 0), None);
    }

    #[test]
    fn test_parse_dict_operand_single_byte_zero() {
        // b0 = 139 => value = 139 - 139 = 0
        assert_eq!(parse_dict_operand(&[139], 0), Some((0, 1)));
    }

    #[test]
    fn test_parse_dict_operand_single_byte_positive() {
        // b0 = 246 => value = 246 - 139 = 107
        assert_eq!(parse_dict_operand(&[246], 0), Some((107, 1)));
    }

    #[test]
    fn test_parse_dict_operand_single_byte_negative() {
        // b0 = 32 => value = 32 - 139 = -107
        assert_eq!(parse_dict_operand(&[32], 0), Some((-107, 1)));
    }

    #[test]
    fn test_parse_dict_operand_two_byte_positive() {
        // b0=247, b1=0 => (247-247)*256 + 0 + 108 = 108
        assert_eq!(parse_dict_operand(&[247, 0], 0), Some((108, 2)));
        // b0=250, b1=255 => (250-247)*256 + 255 + 108 = 1131
        assert_eq!(parse_dict_operand(&[250, 255], 0), Some((1131, 2)));
    }

    #[test]
    fn test_parse_dict_operand_two_byte_negative() {
        // b0=251, b1=0 => -(251-251)*256 - 0 - 108 = -108
        assert_eq!(parse_dict_operand(&[251, 0], 0), Some((-108, 2)));
        // b0=254, b1=255 => -(254-251)*256 - 255 - 108 = -1131
        assert_eq!(parse_dict_operand(&[254, 255], 0), Some((-1131, 2)));
    }

    #[test]
    fn test_parse_dict_operand_two_byte_truncated() {
        // b0=247 but no b1
        assert_eq!(parse_dict_operand(&[247], 0), None);
        assert_eq!(parse_dict_operand(&[251], 0), None);
    }

    #[test]
    fn test_parse_dict_operand_three_byte_int16() {
        // b0=28, then 16-bit signed int
        // 0x00, 0x01 => 1
        assert_eq!(parse_dict_operand(&[28, 0x00, 0x01], 0), Some((1, 3)));
        // 0xFF, 0xFF => -1
        assert_eq!(parse_dict_operand(&[28, 0xFF, 0xFF], 0), Some((-1, 3)));
        // 0x7F, 0xFF => 32767
        assert_eq!(parse_dict_operand(&[28, 0x7F, 0xFF], 0), Some((32767, 3)));
    }

    #[test]
    fn test_parse_dict_operand_three_byte_truncated() {
        assert_eq!(parse_dict_operand(&[28, 0x00], 0), None);
    }

    #[test]
    fn test_parse_dict_operand_five_byte_int32() {
        // b0=29, then 32-bit signed int
        // 0x00, 0x00, 0x00, 0x01 => 1
        assert_eq!(parse_dict_operand(&[29, 0x00, 0x00, 0x00, 0x01], 0), Some((1, 5)));
        // 0xFF, 0xFF, 0xFF, 0xFF => -1
        assert_eq!(parse_dict_operand(&[29, 0xFF, 0xFF, 0xFF, 0xFF], 0), Some((-1, 5)));
    }

    #[test]
    fn test_parse_dict_operand_five_byte_truncated() {
        assert_eq!(parse_dict_operand(&[29, 0x00, 0x00, 0x00], 0), None);
    }

    #[test]
    fn test_parse_dict_operand_real_number() {
        // b0=30, then nibble pairs terminated by 0xF
        // 1.5 would be encoded as: 0x1A 0x5F (1, '.', 5, end)
        // But since we just skip real numbers and return 0...
        let data = [30, 0x1A, 0x5F];
        let result = parse_dict_operand(&data, 0);
        assert!(result.is_some());
        let (val, consumed) = result.unwrap();
        assert_eq!(val, 0); // Real numbers return 0
        assert_eq!(consumed, 3);
    }

    #[test]
    fn test_parse_dict_operand_real_nibble1_end() {
        // nibble1 = 0xF => end marker in high nibble
        let data = [30, 0xF0];
        let result = parse_dict_operand(&data, 0);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), (0, 2));
    }

    #[test]
    fn test_parse_dict_operand_real_unterminated() {
        // Real number that runs off the end without terminator
        let data = [30, 0x12, 0x34];
        assert_eq!(parse_dict_operand(&data, 0), None);
    }

    #[test]
    fn test_parse_dict_operand_unknown_byte() {
        // Byte values 0-21 are operators, not operands => None
        assert_eq!(parse_dict_operand(&[0], 0), None);
        assert_eq!(parse_dict_operand(&[21], 0), None);
        // Byte 255 should be handled by the 251..=254 range => this is 255
        assert_eq!(parse_dict_operand(&[255], 0), None);
    }

    #[test]
    fn test_parse_dict_operand_with_offset() {
        // Parse from a non-zero position
        let data = [0x00, 0x00, 139]; // value 0 at position 2
        assert_eq!(parse_dict_operand(&data, 2), Some((0, 1)));
    }

    // ==========================================
    // parse_top_dict tests
    // ==========================================

    #[test]
    fn test_parse_top_dict_empty() {
        let (enc, charset) = parse_top_dict(&[]);
        assert_eq!(enc, 0);
        assert_eq!(charset, 0);
    }

    #[test]
    fn test_parse_top_dict_encoding_offset() {
        // Push operand 42 (139+42=181), then operator 16 (encoding)
        let data = [181, 16];
        let (enc, charset) = parse_top_dict(&data);
        assert_eq!(enc, 42);
        assert_eq!(charset, 0); // default
    }

    #[test]
    fn test_parse_top_dict_charset_offset() {
        // Push operand 99 (139+99=238), then operator 15 (charset)
        let data = [238, 15];
        let (enc, charset) = parse_top_dict(&data);
        assert_eq!(enc, 0); // default
        assert_eq!(charset, 99);
    }

    #[test]
    fn test_parse_top_dict_both_offsets() {
        // encoding=50 (189), op 16, charset=100 (239), op 15
        let data = [189, 16, 239, 15];
        let (enc, charset) = parse_top_dict(&data);
        assert_eq!(enc, 50);
        assert_eq!(charset, 100);
    }

    #[test]
    fn test_parse_top_dict_two_byte_operator() {
        // b0=12 signals two-byte operator; b0=12, b1=X => operator (12<<8)|X
        // e.g. operator 12, sub 0 => (3072) => not encoding/charset, should be ignored
        let data = [139, 12, 0]; // push 0, then op 12.0
        let (enc, charset) = parse_top_dict(&data);
        assert_eq!(enc, 0);
        assert_eq!(charset, 0);
    }

    #[test]
    fn test_parse_top_dict_unknown_operator() {
        // Unknown operator 17 (not encoding=16 or charset=15)
        let data = [181, 17]; // push 42, operator 17
        let (enc, charset) = parse_top_dict(&data);
        assert_eq!(enc, 0); // not set
        assert_eq!(charset, 0); // not set
    }

    #[test]
    fn test_parse_top_dict_skip_unparseable() {
        // If operand parse fails, we just skip the byte
        // Use byte 255 which is not a valid operand (nor operator)
        let data = [255, 181, 16]; // 255 skipped, then 42 + encoding op
        let (enc, charset) = parse_top_dict(&data);
        assert_eq!(enc, 42);
        assert_eq!(charset, 0);
    }

    // ==========================================
    // parse_charset tests
    // ==========================================

    #[test]
    fn test_parse_charset_out_of_bounds() {
        assert_eq!(parse_charset(&[0x00], 5, 10), None);
    }

    #[test]
    fn test_parse_charset_format0() {
        // format=0, then SIDs for GID 1,2,3...
        let data = vec![
            0x00, // format 0
            0x00, 0x01, // SID 1 (space)
            0x00, 0x22, // SID 34 (A)
            0x00, 0x42, // SID 66 (a)
        ];
        let result = parse_charset(&data, 0, 4);
        assert!(result.is_some());
        let sids = result.unwrap();
        assert_eq!(sids.len(), 4);
        assert_eq!(sids[0], 0); // .notdef
        assert_eq!(sids[1], 1); // space
        assert_eq!(sids[2], 34); // A
        assert_eq!(sids[3], 66); // a
    }

    #[test]
    fn test_parse_charset_format1() {
        // format=1, range: first_sid=34, n_left=2 => SIDs 34,35,36
        let data = vec![
            0x01, // format 1
            0x00, 0x22, // first_sid = 34
            0x02, // n_left = 2
        ];
        let result = parse_charset(&data, 0, 4);
        assert!(result.is_some());
        let sids = result.unwrap();
        assert_eq!(sids.len(), 4);
        assert_eq!(sids[0], 0); // .notdef
        assert_eq!(sids[1], 34); // A
        assert_eq!(sids[2], 35); // B
        assert_eq!(sids[3], 36); // C
    }

    #[test]
    fn test_parse_charset_format2() {
        // format=2, range: first_sid=66, n_left=3 => SIDs 66,67,68,69
        let data = vec![
            0x02, // format 2
            0x00, 0x42, // first_sid = 66
            0x00, 0x03, // n_left = 3
        ];
        let result = parse_charset(&data, 0, 5);
        assert!(result.is_some());
        let sids = result.unwrap();
        assert_eq!(sids.len(), 5);
        assert_eq!(sids[0], 0);
        assert_eq!(sids[1], 66); // a
        assert_eq!(sids[2], 67); // b
        assert_eq!(sids[3], 68); // c
        assert_eq!(sids[4], 69); // d
    }

    #[test]
    fn test_parse_charset_unknown_format() {
        let data = vec![0x03]; // format 3 doesn't exist
        assert_eq!(parse_charset(&data, 0, 2), None);
    }

    #[test]
    fn test_parse_charset_format0_truncated() {
        // format=0 but not enough data for SIDs
        let data = vec![0x00, 0x00]; // only 1 byte of SID data (need 2)
        let result = parse_charset(&data, 0, 3);
        assert!(result.is_some());
        let sids = result.unwrap();
        // Should get .notdef + whatever it could parse
        assert_eq!(sids[0], 0);
    }

    #[test]
    fn test_parse_charset_format1_limits_to_num_glyphs() {
        // Range would give more SIDs than needed
        let data = vec![
            0x01, // format 1
            0x00, 0x01, // first_sid = 1
            0xFF, // n_left = 255 (way more than we need)
        ];
        let result = parse_charset(&data, 0, 3);
        assert!(result.is_some());
        let sids = result.unwrap();
        assert_eq!(sids.len(), 3); // limited to num_glyphs
    }

    // ==========================================
    // parse_encoding_table tests
    // ==========================================

    #[test]
    fn test_parse_encoding_table_out_of_bounds() {
        assert_eq!(parse_encoding_table(&[0x00], 5), None);
    }

    #[test]
    fn test_parse_encoding_table_format0() {
        let data = vec![
            0x00, // format 0 (no supplement)
            0x03, // n_codes = 3
            0x41, // code 0x41 => GID 1
            0x42, // code 0x42 => GID 2
            0x43, // code 0x43 => GID 3
        ];
        let result = parse_encoding_table(&data, 0);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map.get(&0x41), Some(&1));
        assert_eq!(map.get(&0x42), Some(&2));
        assert_eq!(map.get(&0x43), Some(&3));
    }

    #[test]
    fn test_parse_encoding_table_format1() {
        let data = vec![
            0x01, // format 1
            0x01, // n_ranges = 1
            0x41, // first = 0x41
            0x02, // n_left = 2 (codes 0x41, 0x42, 0x43)
        ];
        let result = parse_encoding_table(&data, 0);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map.get(&0x41), Some(&1));
        assert_eq!(map.get(&0x42), Some(&2));
        assert_eq!(map.get(&0x43), Some(&3));
    }

    #[test]
    fn test_parse_encoding_table_unknown_format() {
        let data = vec![0x02]; // format 2 doesn't exist for encoding
                               // 0x02 & 0x7F = 2
        assert_eq!(parse_encoding_table(&data, 0), None);
    }

    #[test]
    fn test_parse_encoding_table_format0_truncated() {
        let data = vec![
            0x00, // format 0
            0x05, // n_codes = 5 but only 2 bytes of data follow
            0x41, 0x42,
        ];
        let result = parse_encoding_table(&data, 0);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_parse_encoding_table_with_supplement() {
        let data = vec![
            0x80, // format 0 with supplement flag (bit 7 set)
            0x01, // n_codes = 1
            0x41, // code 0x41 => GID 1
            0x01, // n_sups = 1
            0x42, // supplement code = 0x42
            0x00, 0x22, // supplement SID = 34
        ];
        let result = parse_encoding_table(&data, 0);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map.get(&0x41), Some(&1));
        assert_eq!(map.get(&0x42), Some(&34)); // supplement
    }

    #[test]
    fn test_parse_encoding_table_format1_truncated() {
        let data = vec![
            0x01, // format 1
            0x02, // n_ranges = 2, but only 1 range of data follows
            0x41, 0x01, // range 1: first=0x41, n_left=1
        ];
        let result = parse_encoding_table(&data, 0);
        assert!(result.is_some());
        let map = result.unwrap();
        assert_eq!(map.len(), 2); // Only first range parsed
    }

    #[test]
    fn test_parse_encoding_table_format0_empty_pos() {
        // format 0, but pos after format byte is at end
        let data = vec![0x00];
        assert_eq!(parse_encoding_table(&data, 0), None);
    }

    #[test]
    fn test_parse_encoding_table_format1_empty_pos() {
        let data = vec![0x01];
        assert_eq!(parse_encoding_table(&data, 0), None);
    }

    // ==========================================
    // resolve_glyph_name tests
    // ==========================================

    #[test]
    fn test_resolve_glyph_name_predefined() {
        let string_index: Vec<&[u8]> = vec![];
        assert_eq!(resolve_glyph_name(0, &string_index), Some(".notdef".to_string()));
        assert_eq!(resolve_glyph_name(1, &string_index), Some("space".to_string()));
        assert_eq!(resolve_glyph_name(34, &string_index), Some("A".to_string()));
        assert_eq!(resolve_glyph_name(390, &string_index), Some("Semibold".to_string()));
    }

    #[test]
    fn test_resolve_glyph_name_custom_string() {
        let custom: Vec<&[u8]> = vec![b"MyGlyph", b"AnotherGlyph"];
        // SID 391 => index 0 in string_index
        assert_eq!(resolve_glyph_name(391, &custom), Some("MyGlyph".to_string()));
        // SID 392 => index 1
        assert_eq!(resolve_glyph_name(392, &custom), Some("AnotherGlyph".to_string()));
    }

    #[test]
    fn test_resolve_glyph_name_custom_out_of_range() {
        let custom: Vec<&[u8]> = vec![b"OnlyOne"];
        assert_eq!(resolve_glyph_name(393, &custom), None); // index 2, but only 1 entry
    }

    #[test]
    fn test_resolve_glyph_name_custom_invalid_utf8() {
        let invalid_utf8: Vec<&[u8]> = vec![&[0xFF, 0xFE]];
        assert_eq!(resolve_glyph_name(391, &invalid_utf8), None);
    }

    // ==========================================
    // extract_cff_from_opentype tests
    // ==========================================

    #[test]
    fn test_extract_cff_from_opentype_too_short() {
        assert_eq!(extract_cff_from_opentype(&[0; 8]), None);
    }

    #[test]
    fn test_extract_cff_from_opentype_not_opentype() {
        let data = vec![0x00; 16];
        assert_eq!(extract_cff_from_opentype(&data), None);
    }

    #[test]
    fn test_extract_cff_from_opentype_otto_no_cff_table() {
        // "OTTO" magic, 0 tables
        let data = vec![
            0x4F, 0x54, 0x54, 0x4F, // "OTTO"
            0x00, 0x00, // num_tables = 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // searchRange, entrySelector, rangeShift
        ];
        assert_eq!(extract_cff_from_opentype(&data), None);
    }

    #[test]
    fn test_extract_cff_from_opentype_with_cff_table() {
        let cff_data = b"\x01\x00\x04\x01"; // Minimal CFF header
        let cff_offset: u32 = 28; // 12 (header) + 16 (one table record)
        let cff_length: u32 = cff_data.len() as u32;

        let mut data = vec![
            0x4F, 0x54, 0x54, 0x4F, // "OTTO"
            0x00, 0x01, // num_tables = 1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // searchRange etc
        ];
        // Table record: tag "CFF ", checksum, offset, length
        data.extend_from_slice(b"CFF "); // tag
        data.extend_from_slice(&[0, 0, 0, 0]); // checksum
        data.extend_from_slice(&cff_offset.to_be_bytes()); // offset
        data.extend_from_slice(&cff_length.to_be_bytes()); // length
        data.extend_from_slice(cff_data);

        let result = extract_cff_from_opentype(&data);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), cff_data);
    }

    #[test]
    fn test_extract_cff_from_opentype_truncated_table_dir() {
        // OTTO with 1 table but data too short
        let data = vec![
            0x4F, 0x54, 0x54, 0x4F, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            // table record starts but is truncated
            b'C', b'F',
        ];
        assert_eq!(extract_cff_from_opentype(&data), None);
    }

    // ==========================================
    // parse_cff_encoding (integration) tests
    // ==========================================

    #[test]
    fn test_parse_cff_encoding_too_short() {
        assert_eq!(parse_cff_encoding(&[0, 1, 2]), None);
    }

    #[test]
    fn test_parse_cff_encoding_wrong_version() {
        // Not version 1, and not an OpenType wrapper
        let data = vec![0x02, 0x00, 0x04, 0x01, 0x00];
        assert_eq!(parse_cff_encoding(&data), None);
    }

    #[test]
    fn test_parse_cff_encoding_version1_too_short_after_check() {
        // Version byte is 1 but too small overall
        let data = vec![0x01, 0x00, 0x04];
        assert_eq!(parse_cff_encoding(&data), None);
    }

    #[test]
    fn test_parse_cff_encoding_expert_encoding() {
        // Build a minimal valid CFF with encoding_offset=1 (ExpertEncoding)
        // This requires: header, name INDEX, top dict INDEX (with encoding=1), string INDEX
        let data = build_minimal_cff(1, 0);
        let result = parse_cff_encoding(&data);
        assert_eq!(result, None); // ExpertEncoding => None
    }

    #[test]
    fn test_parse_cff_encoding_standard_encoding_default_charset() {
        // encoding_offset=0, charset_offset=0 (both defaults)
        let data = build_minimal_cff(0, 0);
        let result = parse_cff_encoding(&data);
        assert_eq!(result, None); // StandardEncoding with default charset => None
    }

    /// Helper: builds a minimal CFF font binary with specified encoding and charset offsets.
    fn build_minimal_cff(encoding_offset: i32, charset_offset: i32) -> Vec<u8> {
        // CFF Header: major=1, minor=0, hdrSize=4, offSize=1
        let mut data = vec![1, 0, 4, 1];

        // Name INDEX: 1 entry "Test"
        append_index(&mut data, &[b"Test"]);

        // Top DICT INDEX: encode encoding_offset and charset_offset
        let top_dict = build_top_dict(encoding_offset, charset_offset);
        append_index(&mut data, &[&top_dict]);

        // String INDEX: empty
        append_index(&mut data, &[]);

        // Global Subr INDEX: empty
        append_index(&mut data, &[]);

        data
    }

    /// Encode a CFF DICT with encoding (op 16) and charset (op 15) operands.
    fn build_top_dict(encoding_offset: i32, charset_offset: i32) -> Vec<u8> {
        let mut dict = Vec::new();
        // Encode encoding_offset as operand, then op 16
        encode_dict_int(&mut dict, encoding_offset);
        dict.push(16); // encoding operator
                       // Encode charset_offset as operand, then op 15
        encode_dict_int(&mut dict, charset_offset);
        dict.push(15); // charset operator
        dict
    }

    /// Encode a CFF integer operand into DICT format.
    fn encode_dict_int(out: &mut Vec<u8>, val: i32) {
        if (-107..=107).contains(&val) {
            out.push((val + 139) as u8);
        } else if (108..=1131).contains(&val) {
            let v = val - 108;
            out.push((v / 256 + 247) as u8);
            out.push((v % 256) as u8);
        } else if (-1131..=-108).contains(&val) {
            let v = -val - 108;
            out.push((v / 256 + 251) as u8);
            out.push((v % 256) as u8);
        } else if (-32768..=32767).contains(&val) {
            out.push(28);
            let bytes = (val as i16).to_be_bytes();
            out.push(bytes[0]);
            out.push(bytes[1]);
        } else {
            out.push(29);
            let bytes = val.to_be_bytes();
            out.extend_from_slice(&bytes);
        }
    }

    /// Append a CFF INDEX to a data vector.
    fn append_index(data: &mut Vec<u8>, entries: &[&[u8]]) {
        let count = entries.len() as u16;
        data.extend_from_slice(&count.to_be_bytes());
        if count == 0 {
            return;
        }
        data.push(1); // off_size = 1

        // Offsets (1-based)
        let mut offset: u8 = 1;
        data.push(offset);
        for entry in entries {
            offset += entry.len() as u8;
            data.push(offset);
        }
        // Data
        for entry in entries {
            data.extend_from_slice(entry);
        }
    }

    // ==========================================
    // glyph_name_to_sid tests
    // ==========================================

    #[test]
    fn test_glyph_name_to_sid_known_names() {
        assert_eq!(glyph_name_to_sid(".notdef"), Some(0));
        assert_eq!(glyph_name_to_sid("space"), Some(1));
        assert_eq!(glyph_name_to_sid("A"), Some(34));
        assert_eq!(glyph_name_to_sid("B"), Some(35));
        assert_eq!(glyph_name_to_sid("Z"), Some(59));
        assert_eq!(glyph_name_to_sid("a"), Some(66));
        assert_eq!(glyph_name_to_sid("z"), Some(91));
        assert_eq!(glyph_name_to_sid("zero"), Some(17));
        assert_eq!(glyph_name_to_sid("nine"), Some(26));
    }

    #[test]
    fn test_glyph_name_to_sid_unknown() {
        assert_eq!(glyph_name_to_sid("nonexistent_glyph_xyz"), None);
        assert_eq!(glyph_name_to_sid(""), None);
    }

    #[test]
    fn test_glyph_name_to_sid_roundtrip() {
        // Verify sid_to_name and glyph_name_to_sid are consistent
        for sid in 0u16..391 {
            if let Some(name) = sid_to_name(sid) {
                assert_eq!(
                    glyph_name_to_sid(name),
                    Some(sid),
                    "Roundtrip failed for SID {} (name '{}')",
                    sid,
                    name
                );
            }
        }
    }

    // ==========================================
    // parse_cff_gid_mapping tests
    // ==========================================

    #[test]
    fn test_parse_cff_gid_mapping_invalid_data() {
        assert!(parse_cff_gid_mapping(&[]).is_none());
        assert!(parse_cff_gid_mapping(&[0, 1, 2]).is_none());
        assert!(parse_cff_gid_mapping(&[2, 0, 4, 2]).is_none()); // wrong version
    }
}
