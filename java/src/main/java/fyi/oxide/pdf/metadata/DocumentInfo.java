/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */
package fyi.oxide.pdf.metadata;

import java.util.Optional;
import org.jspecify.annotations.Nullable;

/**
 * The PDF Info dictionary: title, author, subject, keywords, creator,
 * producer, creation/modification dates. Encoded in PDFDocEncoding or
 * UTF-16; pdf_oxide normalizes both to Java {@code String}.
 */
public final class DocumentInfo {

    private final @Nullable String title;
    private final @Nullable String author;
    private final @Nullable String subject;
    private final @Nullable String keywords;
    private final @Nullable String creator;
    private final @Nullable String producer;
    private final @Nullable String creationDate;
    private final @Nullable String modificationDate;

    public DocumentInfo(
            @Nullable String title,
            @Nullable String author,
            @Nullable String subject,
            @Nullable String keywords,
            @Nullable String creator,
            @Nullable String producer,
            @Nullable String creationDate,
            @Nullable String modificationDate) {
        this.title = title;
        this.author = author;
        this.subject = subject;
        this.keywords = keywords;
        this.creator = creator;
        this.producer = producer;
        this.creationDate = creationDate;
        this.modificationDate = modificationDate;
    }

    public Optional<String> title() {
        return Optional.ofNullable(title);
    }

    public Optional<String> author() {
        return Optional.ofNullable(author);
    }

    public Optional<String> subject() {
        return Optional.ofNullable(subject);
    }

    public Optional<String> keywords() {
        return Optional.ofNullable(keywords);
    }

    public Optional<String> creator() {
        return Optional.ofNullable(creator);
    }

    public Optional<String> producer() {
        return Optional.ofNullable(producer);
    }
    /** @return ISO 8601-formatted creation date string, if present. */
    public Optional<String> creationDate() {
        return Optional.ofNullable(creationDate);
    }
    /** @return ISO 8601-formatted modification date string, if present. */
    public Optional<String> modificationDate() {
        return Optional.ofNullable(modificationDate);
    }
}
