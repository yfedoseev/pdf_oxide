<?php

declare(strict_types=1);

namespace PdfOxide\Types;

/**
 * PDF document metadata.
 */
readonly class Metadata
{
    public function __construct(
        public ?string $title = null,
        public ?string $author = null,
        public ?string $subject = null,
        public ?string $keywords = null,
        public ?string $creator = null,
        public ?string $producer = null,
        public ?\DateTime $creationDate = null,
        public ?\DateTime $modificationDate = null,
        public array $customMetadata = []
    ) {}

    /**
     * Check if metadata is empty.
     */
    public function isEmpty(): bool
    {
        return $this->title === null
            && $this->author === null
            && $this->subject === null
            && $this->keywords === null
            && $this->creator === null
            && $this->producer === null
            && $this->creationDate === null
            && $this->modificationDate === null
            && empty($this->customMetadata);
    }

    /**
     * Get all set metadata fields.
     */
    public function getAll(): array
    {
        $metadata = [];

        if ($this->title !== null) {
            $metadata['title'] = $this->title;
        }
        if ($this->author !== null) {
            $metadata['author'] = $this->author;
        }
        if ($this->subject !== null) {
            $metadata['subject'] = $this->subject;
        }
        if ($this->keywords !== null) {
            $metadata['keywords'] = $this->keywords;
        }
        if ($this->creator !== null) {
            $metadata['creator'] = $this->creator;
        }
        if ($this->producer !== null) {
            $metadata['producer'] = $this->producer;
        }
        if ($this->creationDate !== null) {
            $metadata['creation_date'] = $this->creationDate->format(\DateTime::ISO8601);
        }
        if ($this->modificationDate !== null) {
            $metadata['modification_date'] = $this->modificationDate->format(\DateTime::ISO8601);
        }

        return array_merge($metadata, $this->customMetadata);
    }

    /**
     * Convert to array format.
     */
    public function toArray(): array
    {
        return $this->getAll();
    }
}
