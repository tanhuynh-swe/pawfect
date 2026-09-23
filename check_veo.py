"""Check that Veo can actually be called, without generating anything.

Video generation is billed per second, so this answers the free questions
first. Listing the models is not enough: the Gemini API's free tier lists
Veo and then refuses every request with 429, so on that route this says so
plainly rather than calling it ready.
"""
import sys
sys.path.insert(0, "/Users/tanhuynh/Projects/pawfect")
from pipeline.config import load_config, _load_dotenv
from pipeline import veo

_load_dotenv()
settings = load_config()["visuals"].get("veo") or {}
try:
    client = veo._client(settings)
except Exception as exc:
    raise SystemExit(f"no client: {exc}")

model = settings.get("model", veo.MODEL_DEFAULT)
if client.vertexai:
    vertex = settings.get("vertex") or {}
    print(f"route: Vertex AI, project {vertex.get('project')!r}, "
          f"{vertex.get('location') or 'us-central1'}")
    try:
        client.models.get(model=model)
    except Exception as exc:
        raise SystemExit(f"  -> {model} not reachable: {exc}\n"
                         "     Enable the Vertex AI API on the project and run\n"
                         "     `gcloud auth application-default login`.")
    print(f"  -> {model} reachable. Quota is only proven by one real clip:\n"
          "     build with max_clips_per_build: 1.")
else:
    print("route: Gemini API key")
    names = [m.name for m in client.models.list() if "veo" in m.name.lower()]
    print(f"  veo models listed: {names or 'NONE'}")
    print("  -> listed is not callable: the free tier has no Veo quota and\n"
          "     returns 429. Set visuals.veo.vertex.project to use a Cloud\n"
          "     free-trial project instead.")
