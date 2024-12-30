import sys
import os
import flask
import base64
import tempfile
import traceback
import torch
from flask import Flask, Response, request, stream_with_context

try:
    from litgpt.generate.base import next_token_image_batch
except ImportError:
    print("Warning: next_token_image_batch not found in litgpt.generate.base.")
    # Define a placeholder function if it's unavailable
    def next_token_image_batch(*args, **kwargs):
        raise NotImplementedError(
            "The function next_token_image_batch is unavailable in this environment."
        )

try:
    from inference_vision import OmniVisionInference
except ImportError as e:
    print(f"Error importing OmniVisionInference: {e}")
    raise

class OmniChatServer:
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device=None) -> None:
        # Check CUDA availability and set device
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.server = Flask(__name__)
        
        try:
            self.client = OmniVisionInference(ckpt_dir, self.device)
            self.client.warm_up()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

        @self.server.route("/", methods=["GET"])
        def index():
            return "Welcome to the OmniChat Server!"

        @self.server.route("/favicon.ico")
        def favicon():
            return "", 204

        @self.server.route("/chat", methods=["POST"])
        def chat():
            try:
                req_data = request.get_json()
                if not req_data or "audio" not in req_data:
                    return Response("Invalid request. 'audio' key is required.", status=400)

                # Decode audio data
                audio_data_buf = base64.b64decode(req_data["audio"].encode("utf-8"))
                stream_stride = req_data.get("stream_stride", 4)
                max_tokens = req_data.get("max_tokens", 2048)

                # Decode image data if provided
                image_data_buf = req_data.get("image")
                image_path = None
                if image_data_buf:
                    image_data_buf = base64.b64decode(image_data_buf.encode("utf-8"))

                # Save temporary files
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f, \
                     tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                    audio_f.write(audio_data_buf)
                    audio_path = audio_f.name
                    if image_data_buf:
                        img_f.write(image_data_buf)
                        image_path = img_f.name

                try:
                    # Generate responses
                    if image_path:
                        resp_generator = self.client.run_vision_AA_batch_stream(
                            audio_path, image_path, stream_stride, max_tokens,
                            save_path='./vision_qa_out_cache.wav'
                        )
                    else:
                        resp_generator = self.client.run_AT_batch_stream(
                            audio_path, stream_stride, max_tokens,
                            save_path='./audio_qa_out_cache.wav'
                        )
                    return Response(stream_with_context(self.generator(resp_generator)),
                                    mimetype='multipart/x-mixed-replace; boundary=frame')
                finally:
                    # Clean up temporary files
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                    if image_path and os.path.exists(image_path):
                        os.unlink(image_path)

            except Exception as e:
                print(traceback.format_exc())
                return Response(f"An error occurred: {str(e)}", status=500)

        self.server = self.server
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
