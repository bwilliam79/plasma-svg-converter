"""
Image processing pipeline for CNC plasma cutting SVG generation.
"""
import cv2
import numpy as np
from PIL import Image
import io
import math


class ImageProcessor:

    def load_image_from_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Load image from raw bytes, preserving alpha if present."""
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Could not decode image — unsupported format.")
        return img

    # ── Subject extraction ─────────────────────────────────────────────────────

    def extract_mask(self, img: np.ndarray) -> np.ndarray:
        """Return a binary mask (0/255) of the main subject."""

        # 1. Alpha channel shortcut (transparent PNG)
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
            return mask

        # 2. AI background removal via rembg (optional dependency)
        try:
            from rembg import remove  # type: ignore
            rgb = cv2.cvtColor(img if img.shape[2] == 3 else img[:, :, :3],
                               cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result = remove(pil_img)
            alpha = np.array(result)[:, :, 3]
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
            return mask
        except Exception:
            pass

        # 3. GrabCut fallback
        return self._grabcut_mask(img)

    def _grabcut_mask(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        margin = max(5, int(min(h, w) * 0.08))
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        try:
            cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            return np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
        except Exception:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return m

    # ── Mask cleanup ───────────────────────────────────────────────────────────

    def clean_mask(self, mask: np.ndarray, min_area_ratio: float = 0.003) -> np.ndarray:
        """Remove small noise and close gaps in the mask."""
        h, w = mask.shape
        min_area = h * w * min_area_ratio

        # Morphological cleanup
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)

        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255

        return cleaned

    # ── Interior hole detection ────────────────────────────────────────────────

    def punch_interior_holes(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Recover interior cutout holes that rembg filled in as foreground.

        Key technique: bilateral filter preserves sharp metal→hole edges while
        smoothing texture noise (wood grain, metal surface variation), giving
        dramatically cleaner hole boundaries than operating on raw pixels.
        """
        # ── BGR image ─────────────────────────────────────────────────────────
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            bgr = img[:, :, :3].copy()
        else:
            bgr = img.copy()

        # ── Solid-filled outer boundary ───────────────────────────────────────
        outer_cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        solid = np.zeros_like(mask)
        cv2.drawContours(solid, outer_cnts, -1, 255, cv2.FILLED)

        # ── Background colour stats (from raw pixels) ────────────────────────
        bg_px = bgr[solid == 0].astype(np.float32)
        if bg_px.shape[0] < 100:
            return mask
        bg_mean = bg_px.mean(axis=0)
        bg_std  = np.clip(bg_px.std(axis=0), 8.0, None)

        # ── Bilateral filter: smooth texture, preserve edges ──────────────────
        # Wood grain noise and metal surface variation are smoothed away, but
        # the sharp metal→hole boundary is preserved.  This is the single
        # biggest improvement for clean hole boundaries.
        smooth = cv2.bilateralFilter(bgr, 9, 75, 75)

        # ── Colour distance on the smoothed image ────────────────────────────
        diff      = smooth.astype(np.float32) - bg_mean
        norm_dist = np.sqrt(np.sum((diff / bg_std) ** 2, axis=2))

        # ── Saturation gate (smoothed image) ─────────────────────────────────
        # Metal (even in shadow) has low saturation; background showing through
        # a hole is coloured (warm wood, green grass, etc.).
        hsv     = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        sat_map = hsv[:, :, 1].astype(np.float32)
        bg_sat  = sat_map[solid == 0]
        sat_thr = float(np.percentile(bg_sat, 20))

        # ── Distance transform for edge safety ───────────────────────────────
        dist_edge = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
        interior  = (dist_edge >= 6).astype(np.uint8) * 255

        # ── Adaptive thresholds from actual background distribution ──────────
        bg_dists  = norm_dist[solid == 0]
        seed_thr  = float(np.clip(np.percentile(bg_dists, 85), 0.8, 3.5))
        cand_thr  = float(np.clip(np.percentile(bg_dists, 96), 1.2, 5.5))

        # ── Seeded region growing ─────────────────────────────────────────────
        sat_ok     = sat_map >= sat_thr
        seeds      = ((interior == 255) & (norm_dist < seed_thr) & sat_ok).astype(np.uint8) * 255
        candidates = ((interior == 255) & (norm_dist < cand_thr) & sat_ok).astype(np.uint8) * 255

        k_grow   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        expanded = cv2.dilate(seeds, k_grow)
        hole_img = cv2.bitwise_and(expanded, candidates)

        # Close gaps within each hole
        k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hole_img = cv2.morphologyEx(hole_img, cv2.MORPH_CLOSE, k_close)
        hole_img[solid == 0] = 0

        # ── Size filter with component-level edge guard ──────────────────────
        # Small components near the outer boundary = edge noise → remove.
        # Large components near the boundary = legitimate features → keep.
        h, w = mask.shape
        total_px    = h * w
        min_hole_px = max(30, int(total_px * 0.00015))
        edge_size   = int(total_px * 0.002)           # "small" threshold

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole_img)
        clean_holes = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_hole_px:
                continue
            component = (labels == i)
            # Small component whose closest pixel is within 20 px of the outer
            # boundary → almost certainly edge-transition noise, not a real hole.
            if area < edge_size and dist_edge[component].min() < 20:
                continue
            clean_holes[component] = 255

        # ── Final boundary smooth: Gaussian blur → 50 % threshold ────────────
        blurred     = cv2.GaussianBlur(clean_holes, (13, 13), 4)
        clean_holes = (blurred > 127).astype(np.uint8) * 255
        clean_holes[solid == 0] = 0

        result = mask.copy()
        result[clean_holes == 255] = 0
        return result

    # ── Contour utilities ──────────────────────────────────────────────────────

    def smooth_contour(self, contour: np.ndarray, window: int = 7) -> np.ndarray:
        """
        Circular moving-average smooth of contour point coordinates.
        Removes pixel-level zigzag noise before simplification without
        changing the overall shape of the path.
        """
        pts = contour.reshape(-1, 2).astype(float)
        n = len(pts)
        if n < window * 3:
            return contour
        half = window // 2
        k = np.ones(window) / window
        px = np.concatenate([pts[-half:, 0], pts[:, 0], pts[:half, 0]])
        py = np.concatenate([pts[-half:, 1], pts[:, 1], pts[:half, 1]])
        sx = np.convolve(px, k, mode='valid')[:n]
        sy = np.convolve(py, k, mode='valid')[:n]
        return np.column_stack([sx, sy]).reshape(-1, 1, 2).astype(np.int32)

    def get_contours(self, mask: np.ndarray):
        """Return (contours, hierarchy) using RETR_CCOMP (two-level: outer + holes)."""
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        return contours, hierarchy

    def simplify_contour(self, contour: np.ndarray, epsilon_factor: float) -> np.ndarray:
        """Douglas-Peucker simplification. epsilon_factor is relative to perimeter."""
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, epsilon_factor * perimeter)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        # Ensure minimum 3 points for a valid polygon
        if len(simplified) < 3:
            return contour
        return simplified

    # ── Bridge insertion ───────────────────────────────────────────────────────

    def add_bridges(self, contour: np.ndarray, bridge_width_px: float, num_bridges: int):
        """
        Insert bridge gaps into a contour path.

        Returns:
            path_d (str): SVG path data with bridge gaps
            marker_pts (list): [(x, y), ...] bridge midpoint positions for visualization
        """
        pts = contour.reshape(-1, 2).astype(float)
        n = len(pts)

        if n < 3 or num_bridges < 1:
            return self._pts_to_path(pts, close=True), []

        # Arc-length parameterisation
        diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        close_dist = np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1])
        all_lens = np.append(seg_lens, close_dist)
        cum_len = np.concatenate([[0], np.cumsum(all_lens)])
        total_len = cum_len[-1]

        half_b = bridge_width_px / 2.0

        # Need at least 3× bridge width clearance between bridges
        if total_len < bridge_width_px * num_bridges * 3:
            return self._pts_to_path(pts, close=True), []

        # Bridge centres, offset by half-interval so they don't land at index 0
        bridge_centers = [(i + 0.5) * total_len / num_bridges for i in range(num_bridges)]

        def in_bridge(arc_pos: float) -> bool:
            for bc in bridge_centers:
                d = abs(arc_pos - bc)
                d = min(d, total_len - d)
                if d < half_b:
                    return True
            return False

        # Build marker points (closest contour point to each bridge centre)
        marker_pts = []
        for bc in bridge_centers:
            idx = int(np.searchsorted(cum_len, bc % total_len)) - 1
            idx = max(0, min(idx, n - 1))
            marker_pts.append((float(pts[idx, 0]), float(pts[idx, 1])))

        # Split contour into continuous segments, skipping bridge zones
        segments: list[list] = []
        current: list = []

        for i in range(n):
            if not in_bridge(cum_len[i]):
                current.append(pts[i])
            else:
                if len(current) >= 2:
                    segments.append(current)
                current = []

        # Handle wrap-around: merge trailing segment with leading if both are uncut
        if current:
            if segments and not in_bridge(cum_len[0]):
                segments[0] = current + segments[0]
            elif len(current) >= 2:
                segments.append(current)

        if not segments:
            return self._pts_to_path(pts, close=True), marker_pts

        path_d = " ".join(
            self._pts_to_path(np.array(seg), close=False)
            for seg in segments
            if len(seg) >= 2
        )
        return path_d, marker_pts

    # ── SVG path helpers ───────────────────────────────────────────────────────

    def _pts_to_path(self, pts: np.ndarray, close: bool = True) -> str:
        """Convert Nx2 array of points to SVG path data string."""
        if len(pts) < 2:
            return ""
        d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"
        for p in pts[1:]:
            d += f" L {p[0]:.1f},{p[1]:.1f}"
        if close:
            d += " Z"
        return d

    # ── Preview image ──────────────────────────────────────────────────────────

    def build_preview(self, img: np.ndarray, contours, hierarchy,
                      bridge_markers: list) -> str:
        """
        Draw contours + bridge markers on a copy of the image.
        Returns base64-encoded PNG data URI.
        """
        import base64

        if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            vis = img[:, :, :3].copy()
        else:
            vis = img.copy()

        # Dim the background
        vis = (vis * 0.4).astype(np.uint8)

        if hierarchy is not None:
            h = hierarchy[0]
            for i, c in enumerate(contours):
                if cv2.contourArea(c) < 50:
                    continue
                is_hole = h[i][3] != -1
                color = (80, 80, 255) if is_hole else (80, 200, 80)
                cv2.drawContours(vis, [c], -1, color, 2)

        for (x, y) in bridge_markers:
            cx, cy = int(x), int(y)
            cv2.circle(vis, (cx, cy), 10, (0, 255, 200), -1)
            cv2.circle(vis, (cx, cy), 12, (255, 255, 255), 2)

        _, buf = cv2.imencode('.png', vis)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def process(self, img_bytes: bytes, options: dict) -> dict:
        """
        Full pipeline: image bytes → SVG string + preview image.

        options keys:
            bridge_width_mm  (float, default 3.0)
            num_bridges      (int,   default 3)
            simplification   (float, 1–10, default 2.0)
            output_width_mm  (float, default 200.0)
            keep_holes       (bool,  default True)
        """
        bridge_width_mm = float(options.get('bridge_width_mm', 3.0))
        num_bridges     = max(0, int(options.get('num_bridges', 3)))
        simplification  = float(options.get('simplification', 2.0))
        output_width_mm = float(options.get('output_width_mm', 200.0))
        keep_holes      = bool(options.get('keep_holes', True))

        # ── Load & optionally downscale for processing ──────────────────────
        img = self.load_image_from_bytes(img_bytes)
        orig_h, orig_w = img.shape[:2]

        max_proc_dim = 1500
        scale = 1.0
        if max(orig_h, orig_w) > max_proc_dim:
            scale = max_proc_dim / max(orig_h, orig_w)
            proc_img = cv2.resize(
                img,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=cv2.INTER_AREA
            )
        else:
            proc_img = img

        # ── Extract mask ────────────────────────────────────────────────────
        mask = self.extract_mask(proc_img)
        mask = self.clean_mask(mask)
        mask = self.punch_interior_holes(proc_img, mask)

        # ── Find contours (in proc_img coordinate space) ────────────────────
        contours, hierarchy = self.get_contours(mask)

        if not contours or hierarchy is None:
            raise ValueError("No subject detected — try adjusting the image or settings.")

        # ── Derived units ───────────────────────────────────────────────────
        mm_per_px       = output_width_mm / orig_w          # mm per original pixel
        svg_h_mm        = orig_h * mm_per_px
        # Bridge width in original-resolution pixels, then scaled to proc-res
        bridge_px_orig  = bridge_width_mm / mm_per_px
        bridge_px_proc  = bridge_px_orig * scale
        eps_factor      = simplification * 0.0006            # tuned for typical shapes

        # ── Build SVG paths ─────────────────────────────────────────────────
        svg_paths: list[str] = []
        all_markers: list    = []
        hier = hierarchy[0]

        stats = {'outer': 0, 'holes': 0, 'bridges': 0}

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50:
                continue

            parent   = hier[i][3]
            is_hole  = parent != -1

            if is_hole and not keep_holes:
                continue

            # Smooth then simplify in proc-image space, then scale to original.
            # Smoothing removes pixel-level zigzag noise from the mask boundary;
            # apply a stronger window for hole contours which tend to be noisier.
            smoothed   = self.smooth_contour(contour, window=9 if is_hole else 5)
            simplified = self.simplify_contour(smoothed, eps_factor)
            if scale != 1.0:
                simplified = (simplified.astype(float) * (1.0 / scale)).astype(np.int32)

            if is_hole:
                path_d, markers = self.add_bridges(simplified, bridge_px_orig, num_bridges)
                if path_d:
                    svg_paths.append(
                        f'<path d="{path_d}" class="cut-path hole"/>'
                    )
                    all_markers.extend(markers)
                    stats['holes'] += 1
                    stats['bridges'] += len(markers)
            else:
                path_d = self._pts_to_path(simplified.reshape(-1, 2), close=True)
                if path_d:
                    svg_paths.append(
                        f'<path d="{path_d}" class="cut-path outer"/>'
                    )
                    stats['outer'] += 1

        if not svg_paths:
            raise ValueError("No usable contours found after processing.")

        # ── Bridge marker circles (for preview overlay in SVG) ──────────────
        markers_svg = "\n    ".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" class="bridge-dot"/>'
            for x, y in all_markers
        )

        # ── Assemble SVG ────────────────────────────────────────────────────
        cut_paths_str = "\n    ".join(svg_paths)
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<!-- Plasma Cutting SVG  |  {output_width_mm}mm × {svg_h_mm:.1f}mm -->
<!-- Outer contours: black  |  Holes: red (with bridge gaps) -->
<svg xmlns="http://www.w3.org/2000/svg"
     width="{output_width_mm}mm" height="{svg_h_mm:.1f}mm"
     viewBox="0 0 {orig_w} {orig_h}">
  <defs>
    <style>
      .cut-path {{ fill: none; stroke-linecap: round; stroke-linejoin: round; }}
      .outer    {{ stroke: #000000; stroke-width: 1.5; }}
      .hole     {{ stroke: #cc0000; stroke-width: 1.5; }}
      .bridge-dot {{ fill: #00e5ff; opacity: 0.85; }}
    </style>
  </defs>
  <g id="cut-layer">
    {cut_paths_str}
  </g>
  <g id="bridge-markers">
    {markers_svg}
  </g>
</svg>'''

        # ── Build preview image ─────────────────────────────────────────────
        preview_img = self.build_preview(proc_img, contours, hierarchy,
                                         [(x * scale, y * scale) for x, y in all_markers])

        return {
            'svg': svg,
            'preview': preview_img,
            'stats': stats,
            'dimensions': {
                'width_mm':  output_width_mm,
                'height_mm': round(svg_h_mm, 1),
            }
        }
