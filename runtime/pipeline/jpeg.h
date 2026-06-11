// pipeline/jpeg.h — minimal libjpeg-turbo wrapper that decodes raw JPEG
// bytes (the file contents) into an RGB888 NHWC buffer + width/height.
// This is relatively stale and used by the PPU path onlt.
//
// Decode produces 8-bit-per-channel RGB in row-major (height-by-width
// rows, channel innermost). The image is stored as
// `bytes[(y * width + x) * 3 + c]`.

#ifndef TIMVX_PIPELINE_JPEG_H
#define TIMVX_PIPELINE_JPEG_H

#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" {
#include <jpeglib.h>
}

namespace timvx_pipeline {

struct RgbImage {
  std::vector<uint8_t> data;  // size = height * width * 3
  int width  = 0;
  int height = 0;
};

namespace detail {
struct JpegErrorMgr {
  jpeg_error_mgr base;
  jmp_buf jmpbuf;
};
inline void jpeg_error_exit(j_common_ptr cinfo) {
  auto* err = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
  char buf[JMSG_LENGTH_MAX] = {0};
  (*cinfo->err->format_message)(cinfo, buf);
  std::fprintf(stderr, "[jpeg] decode error: %s\n", buf);
  std::longjmp(err->jmpbuf, 1);
}
} // namespace detail

// Decode `bytes` (the raw contents of a .jpg file, including the
// FFD8…FFD9 framing) into an RgbImage. Returns an empty RgbImage on
// failure (and prints to stderr).
//
// Color conversion is libjpeg's standard YCbCr→RGB; grayscale JPEGs
// are expanded to 3 identical channels so downstream resize/normalize
// can treat them uniformly.
inline RgbImage decode_jpeg(const std::vector<uint8_t>& bytes) {
  RgbImage img;
  if (bytes.size() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8) {
    std::fprintf(stderr, "[jpeg] missing FFD8 magic; not a JPEG file\n");
    return img;
  }
  jpeg_decompress_struct cinfo{};
  detail::JpegErrorMgr jerr{};
  cinfo.err = jpeg_std_error(&jerr.base);
  jerr.base.error_exit = detail::jpeg_error_exit;
  if (setjmp(jerr.jmpbuf)) {
    jpeg_destroy_decompress(&cinfo);
    img.data.clear();
    img.width = img.height = 0;
    return img;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, bytes.data(), bytes.size());
  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    std::fprintf(stderr, "[jpeg] read_header did not return JPEG_HEADER_OK\n");
    jpeg_destroy_decompress(&cinfo);
    return img;
  }
  // Force RGB output even on grayscale / CMYK / other source colorspaces.
  cinfo.out_color_space = JCS_RGB;
  jpeg_start_decompress(&cinfo);
  img.width  = static_cast<int>(cinfo.output_width);
  img.height = static_cast<int>(cinfo.output_height);
  size_t row_stride = static_cast<size_t>(img.width) * 3;
  img.data.assign(static_cast<size_t>(img.height) * row_stride, 0);

  // Decompress one scanline at a time into the output buffer.
  while (cinfo.output_scanline < cinfo.output_height) {
    uint8_t* row = img.data.data() + cinfo.output_scanline * row_stride;
    JSAMPROW rows[1] = { row };
    jpeg_read_scanlines(&cinfo, rows, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return img;
}

} // namespace timvx_pipeline

#endif // TIMVX_PIPELINE_JPEG_H
