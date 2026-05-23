# Installation Guide

Complete setup guide for PDF Oxide PHP binding.

## Prerequisites

- PHP 8.1 or higher
- Composer package manager
- FFI extension support
- Native PDF Oxide library (libpdf_oxide)

## Step 1: Verify PHP Version

```bash
php -v
# PHP 8.1.0 or higher required
```

## Step 2: Enable FFI Extension

### Linux (Ubuntu/Debian)

```bash
# Install PHP FFI extension
sudo apt-get update
sudo apt-get install php-ffi

# Verify
php -m | grep ffi
# Should output: ffi
```

### Linux (CentOS/RHEL)

```bash
sudo yum install php-pecl-ffi

# Or compile from source
pecl install ffi
```

### macOS

```bash
# Using Homebrew
brew install php-ffi

# Or compile from source
pecl install ffi
```

### Windows

1. Download pre-built PHP binaries with FFI support
2. Or compile PHP with `--enable-ffi` flag
3. Edit `php.ini`:
   ```ini
   extension=ffi
   ```

### Verify FFI is Enabled

```bash
php -r "echo extension_loaded('ffi') ? 'FFI enabled' : 'FFI disabled';"
```

## Step 3: Install the Library

### Option A: Composer (Recommended)

```bash
composer require pdf-oxide/pdf-oxide
```

### Option B: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/anthropics/pdf_oxide.git
cd pdf_oxide/php
```

2. Install dependencies:
```bash
composer install
```

## Step 4: Install Native Library

The native PDF Oxide library must be accessible to PHP.

### Linux Installation

```bash
# Download or build libpdf_oxide.so
# Copy to standard library path
sudo cp libpdf_oxide.so /usr/local/lib/
sudo ldconfig

# Or add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

### macOS Installation

```bash
# Copy to standard library path
cp libpdf_oxide.dylib /usr/local/lib/
# Or
cp libpdf_oxide.dylib /opt/homebrew/lib/

# Verify
otool -L libpdf_oxide.dylib
```

### Windows Installation

```batch
# Copy to a directory in PATH
copy pdf_oxide.dll C:\Windows\System32\
# Or
set PATH=%PATH%;C:\path\to\lib
```

## Step 5: Verify Installation

Create a test file `test-installation.php`:

```php
<?php
require 'vendor/autoload.php';

use PdfOxide\FFI\NativeLibrary;

// Get platform information
$info = NativeLibrary::getPlatformInfo();

echo "=== PDF Oxide Installation Info ===\n";
echo "PHP Version: " . PHP_VERSION . "\n";
echo "Operating System: " . $info['os'] . "\n";
echo "Platform: " . $info['platform'] . "\n";
echo "FFI Available: " . ($info['ffi_available'] ? 'Yes' : 'No') . "\n";

if ($info['ffi_available']) {
    $header = NativeLibrary::getHeaderFile();
    $library = NativeLibrary::getLibraryFile();

    echo "Header File: " . ($header ? $header : 'Not found') . "\n";
    echo "Library File: " . ($library ? $library : 'Not found') . "\n";

    if ($header && $library) {
        echo "\n✅ Installation successful!\n";
    } else {
        echo "\n❌ Header or library file not found.\n";
    }
} else {
    echo "\n❌ FFI extension not available.\n";
}
?>
```

Run it:
```bash
php test-installation.php
```

Expected output:
```
=== PDF Oxide Installation Info ===
PHP Version: 8.1.10
Operating System: Linux
Platform: linux
FFI Available: Yes
Header File: /path/to/include/pdf_oxide.h
Library File: /usr/local/lib/libpdf_oxide.so

✅ Installation successful!
```

## Troubleshooting

### FFI Not Loaded

**Error:** `Fatal error: Class 'FFI' not found`

**Solution:**
```bash
# Check if FFI is installed
php -m | grep ffi

# If not, install it
sudo apt-get install php-ffi  # Ubuntu/Debian

# Check php.ini
php -i | grep "Loaded Configuration File"
cat /path/to/php.ini | grep -i ffi
```

### Library Not Found

**Error:** `RuntimeException: PDF Oxide library not found`

**Solution:**
```bash
# Find the library
find / -name "libpdf_oxide.so" 2>/dev/null

# Add to library path
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH

# Make permanent (add to ~/.bashrc or ~/.profile)
echo 'export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Permission Issues

**Error:** `RuntimeException: Permission denied`

**Solution:**
```bash
# Check file permissions
ls -la /usr/local/lib/libpdf_oxide.so

# Make readable
sudo chmod 644 /usr/local/lib/libpdf_oxide.so
sudo chmod 755 /usr/local/lib
```

### macOS-Specific Issues

**Error:** `dyld: Library not loaded`

**Solution:**
```bash
# Check library dependencies
otool -L libpdf_oxide.dylib

# Update library paths if needed
install_name_tool -change /old/path /new/path libpdf_oxide.dylib
```

## Platform-Specific Setup

### Docker Setup

```dockerfile
FROM php:8.1-cli

# Install FFI extension
RUN apt-get update && apt-get install -y \
    php-dev \
    && pecl install ffi \
    && docker-php-ext-enable ffi

# Copy your project
COPY . /app
WORKDIR /app

# Install dependencies
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer
RUN composer install

# Copy native library
COPY libpdf_oxide.so /usr/local/lib/
RUN ldconfig

CMD ["php", "script.php"]
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pdf-processor
spec:
  containers:
  - name: php
    image: php:8.1-cli
    env:
    - name: LD_LIBRARY_PATH
      value: /libs
    volumeMounts:
    - name: pdf-libs
      mountPath: /libs
  volumes:
  - name: pdf-libs
    hostPath:
      path: /path/to/libs
```

## Environment Variables

Set these to control library loading behavior:

```bash
# Linux
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/path/to/lib:$DYLD_LIBRARY_PATH

# Windows
set PATH=%PATH%;C:\path\to\lib
```

## Next Steps

1. Review the [API Reference](API_REFERENCE.md)
2. Check the [examples](../examples/) directory
3. Read [WooCommerce Integration Guide](WOOCOMMERCE_GUIDE.md)
4. Review [Error Handling](docs/error-handling.md)

## Getting Help

- 📖 [Full Documentation](docs/)
- 🐛 [Report Issues](https://github.com/anthropics/pdf_oxide/issues)
- 💬 [Discussions](https://github.com/anthropics/pdf_oxide/discussions)
