# AI & Machine Learning Projects

LLM applications, recommender systems and computer vision projects from my Master of IT at UNSW.

The two largest projects have their own repositories; the computer vision projects live in this one.

| Project | What it is | Stack | Highlight |
|---|---|---|---|
| [MediChain](https://github.com/421D/medichain-multi-agent) | Multi-agent LLM system for clinical diagnostic reasoning, with a structured Critic ⇄ Diagnostician debate | LangGraph, FastAPI, RAG (ChromaDB), MCP, React | I designed all six agent prompts; Top-1 100% on 7 ground-truth cases |
| [Fashion Recommender](https://github.com/421D/fashion-recommender-cold-start) | Hybrid recommender on 8.5M Amazon reviews with a knowledge-based path for brand-new users | pandas, scikit-learn, Surprise SVD | I built the cold-start module and cut its runtime from 13.2 h to ~15 s |
| [Fashion Image Classification](fashion-image-classification/) | 4-class product photo classification with four pretrained CNNs, tested for robustness to noise | PyTorch, MobileNetV2, ResNet, EfficientNet | EfficientNet-B0 is the most noise-robust (92.4%) with 5.3M parameters |
| [Aerial Scene Classification](aerial-scene-classification/) | 15-class aerial images with fine-tuned ResNet-18 and EfficientNet-B0 | PyTorch, torchvision | EfficientNet-B0 97.99% vs ResNet-18 96.93% (5-fold CV) |

## Repository layout

```
fashion-image-classification/   scripts/, results/, notebook, README
aerial-scene-classification/    scripts/, results/, README
```

Each project folder has its own README with the method, results and how to run it.
