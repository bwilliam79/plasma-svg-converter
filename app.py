"""
Plasma SVG Converter — Flask backend
Run: python app.py  (or: flask run)
"""
from flask import Flask, render_template, request, jsonify, Response
import traceback

from defusedxml import ElementTree as DefusedET
from xml.etree import ElementTree as ET

from processor import ImageProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit

# Upper bound on SVG payload returned from the client for download. The SVG we
# produce is comfortably under 4 MB even for dense shapes; anything larger is
# almost certainly an attempt to abuse the endpoint.
MAX_SVG_BYTES = 8 * 1024 * 1024

SVG_NS = 'http://www.w3.org/2000/svg'


def _sanitize_svg(raw: str) -> str:
    """Parse the client-supplied SVG with a defused XML parser to block XXE
    and external-entity attacks, strip the bridge-markers group (we don't want
    those in the downloaded cut file), and re-serialize.

    Returns the cleaned SVG string. Raises ValueError on invalid input.
    """
    if len(raw.encode('utf-8')) > MAX_SVG_BYTES:
        raise ValueError('SVG payload too large.')

    try:
        # defusedxml disables external entity resolution, DTDs, and network
        # access by default, which is the XXE guarantee we need here.
        root = DefusedET.fromstring(raw)
    except DefusedET.ParseError as e:
        raise ValueError(f'Invalid SVG XML: {e}') from e
    except Exception as e:  # catches defusedxml's EntitiesForbidden etc.
        raise ValueError(f'Rejected SVG content: {e}') from e

    # Reject anything that isn't actually an <svg> root.
    tag = root.tag
    if tag != 'svg' and tag != f'{{{SVG_NS}}}svg':
        raise ValueError('Root element is not <svg>.')

    # Remove the bridge-markers group contents; keep an empty placeholder so
    # the layer structure is preserved for downstream tooling that expects it.
    for parent in root.iter():
        for child in list(parent):
            if child.get('id') == 'bridge-markers':
                for grandchild in list(child):
                    child.remove(grandchild)

    # Ensure the default SVG namespace is preserved on serialization.
    ET.register_namespace('', SVG_NS)
    return ET.tostring(root, encoding='unicode', xml_declaration=True)


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
    """Return a clean cut-only SVG (no bridge markers) as a file download.

    The client posts back the SVG it received from /process. We parse it with
    defusedxml (blocking XXE / external entities) and strip the bridge-marker
    group via an XML tree edit rather than a regex. Regex-based XML rewriting
    is unsafe in general, and for a file that drives a physical cutter we
    can't afford malformed output.
    """
    svg = request.form.get('svg', '')
    if not svg:
        return 'No SVG provided.', 400

    try:
        clean = _sanitize_svg(svg)
    except ValueError as e:
        return str(e), 400

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
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=port)
