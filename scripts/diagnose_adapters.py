"""Diagnostic: check what adapters actually got loaded and what their real
names are, so we can see why set_active_adapters isn't finding them.
"""
from calli.embeddings.specter2 import Specter2Embedder

embedder = Specter2Embedder()

print("Adapters registered on the model:")
print(list(embedder.model.adapters_config.adapters.keys()))

print()
print("Active adapters right after init (should be none, we set_active=False):")
print(embedder.model.active_adapters)

embedder.model.set_active_adapters("query")
print()
print("Active adapters after calling set_active_adapters('query'):")
print(embedder.model.active_adapters)