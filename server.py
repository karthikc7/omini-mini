import sys
import os
import flask
import base64
import tempfile
import traceback
import torch
from flask import Flask, Response, stream_with_context
from inference_vision import OmniVisionInference

class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device=None) -> None:
        # Check CUDA availability and set device
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        server = Flask(__name__)
        
        try:
            self.client = OmniVisionInference(ckpt_dir, self.device)
            self.client.warm_up()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

        @server.route("/", methods=["GET"])
        def index():
            return "Welcome to the OmniChat Server!"

        @server.route("/favicon.ico")
        def favicon():
            return "", 204

        @server.route("/chat", methods=["POST"])
        def chat():
            req_data = flask.request.get_json()
            try:
                audio_data_buf = req_data["audio"].encode("utf-8")
                audio_data_buf = base64.b64decode(audio_data_buf)
                stream_stride = req_data.get("stream_stride", 4)
                max_tokens = req_data.get("max_tokens", 2048)

                image_data_buf = req_data.get("image", None)
                if image_data_buf:
                    image_data_buf = image_data_buf.encode("utf-8")
                    image_data_buf = base64.b64decode(image_data_buf)

                audio_path, img_path = None, None
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f, \
                     tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                    audio_f.write(audio_data_buf)
                    audio_path = audio_f.name

                    if image_data_buf:
                        img_f.write(image_data_buf)
                        img_path = img_f.name
                    else:
                        img_path = None

                    try:
                        if img_path is not None:
                            resp_generator = self.client.run_vision_AA_batch_stream(
                                audio_f.name, img_f.name, stream_stride, max_tokens,
                                save_path='./vision_qa_out_cache.wav'
                            )
                        else:
                            resp_generator = self.client.run_AT_batch_stream(
                                audio_f.name, stream_stride, max_tokens,
                                save_path='./audio_qa_out_cache.wav'
                            )
                        return Response(stream_with_context(self.generator(resp_generator)),
                                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    finally:
                        # Clean up temporary files
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                        if img_path and os.path.exists(img_path):
                            os.unlink(img_path)
            except Exception as e:
                print(traceback.format_exc())
                return Response(f"An error occurred: {str(e)}", status=500)

        self.server = server
        if run_app:
            self.server.run(host=ip, port=port, threaded=False)

    def generator(self, resp_generator):
        for audio_stream, text_stream in resp_generator:
            yield b'\r\n--frame\r\n'
            yield b'Content-Type: audio/wav\r\n\r\n'
            yield audio_stream
            yield b'\r\n--frame\r\n'
            yield b'Content-Type: text/plain\r\n\r\n'
            yield text_stream.encode()

def create_app():
    server = OmniChatServer(run_app=False)
    return server.server

def serve(ip='0.0.0.0', port=60808, device=None):
    OmniChatServer(ip, port=port, run_app=True, device=device)

if __name__ == "__main__":
    import fire
    fire.Fire(serve)