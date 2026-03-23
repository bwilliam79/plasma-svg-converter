"""
Plasma SVG Converter — Flask backend
Run: python app.py  (or: flask run)
"""
from flask import Flask, render_template, request, jsonify, Response
import traceback

from processor import ImageProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    options = {
        'bridge_width_mm': float(request.form.get('bridge_width_mm', 3.0)),
        'num_bridges':     int(request.form.get('num_bridges', 3)),
        'simplification':  float(request.form.get('simplification', 2.0)),
        'output_width_mm': float(request.form.get('output_width_mm', 200.0)),
        'keep_holes':      request.form.get('keep_holes', 'true').lower() == 'true',
    }

    try:
        img_bytes = file.read()
        processor = ImageProcessor()
        result = processor.process(img_bytes, options)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 422
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Processing failed — see server logs.'}), 500


@app.route('/download', methods=['POST'])
def download():
    """Return a clean cut-only SVG (no bridge markers) as a file download."""
    svg = request.form.get('svg', '')
    if not svg:
        return 'No SVG provided.', 400

    # Strip bridge-marker group before download
    import re
    clean = re.sub(
        r'<g id="bridge-markers">.*?</g>',
        '<g id="bridge-markers"/>',
        svg,
        flags=re.DOTALL,
    )

    return Response(
        clean,
        mimetype='image/svg+xml',
        headers={'Content-Disposition': 'attachment; filename="plasma_cut.svg"'},
    )


if __name__ == '__main__':
    import os
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, use_reloader=False, port=port)
