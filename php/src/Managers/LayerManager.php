<?php

declare(strict_types=1);

namespace PdfOxide\Managers;

use FFI\CData;
use PdfOxide\FFI\FunctionBindings;
use PdfOxide\Types\Layer;

/**
 * Manages PDF document layers (Optional Content Groups/OCG).
 *
 * Provides access to layers and visibility settings.
 */
class LayerManager
{
    private FunctionBindings $bindings;
    private CData $handle;
    private ?array $cachedLayers = null;

    public function __construct(CData $handle)
    {
        $this->handle = $handle;
        $this->bindings = new FunctionBindings();
    }

    /**
     * Get total number of layers.
     */
    public function getCount(): int
    {
        return $this->bindings->pdfDocumentGetLayerCount($this->handle);
    }

    /**
     * Get layer at index.
     */
    public function get(int $index): Layer
    {
        if ($this->cachedLayers === null) {
            $this->loadAll();
        }

        if (!isset($this->cachedLayers[$index])) {
            throw new \OutOfRangeException("Layer index $index out of range");
        }

        return $this->cachedLayers[$index];
    }

    /**
     * Get all layers.
     */
    public function getAll(): array
    {
        if ($this->cachedLayers === null) {
            $this->loadAll();
        }
        return $this->cachedLayers;
    }

    /**
     * Load all layers into cache.
     */
    private function loadAll(): void
    {
        $this->cachedLayers = [];
        $count = $this->getCount();

        for ($i = 0; $i < $count; $i++) {
            $name = $this->bindings->pdfDocumentGetLayerName($this->handle, $i);
            $visible = $this->bindings->pdfDocumentIsLayerVisible($this->handle, $i);
            $layer = new Layer($name, $visible);
            $this->cachedLayers[] = $layer;
        }
    }

    /**
     * Check if document has layers.
     */
    public function hasLayers(): bool
    {
        return $this->getCount() > 0;
    }

    /**
     * Get layer names.
     */
    public function getNames(): array
    {
        return array_map(fn(Layer $l) => $l->name, $this->getAll());
    }

    /**
     * Get layer visibility.
     */
    public function getVisibility(): array
    {
        return array_map(fn(Layer $l) => $l->visible, $this->getAll());
    }

    /**
     * Check if layer is visible by name.
     */
    public function isLayerVisible(string $name): bool
    {
        foreach ($this->getAll() as $layer) {
            if ($layer->name === $name) {
                return $layer->visible;
            }
        }
        return false;
    }

    /**
     * Get layers as array.
     */
    public function toArray(): array
    {
        return array_map(fn(Layer $l) => $l->toArray(), $this->getAll());
    }

    /**
     * Clear cached layers.
     *
     * @internal
     */
    public function clearCache(): void
    {
        $this->cachedLayers = null;
    }
}
