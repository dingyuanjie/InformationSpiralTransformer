# Level 6.2 random-marker multi-chunk curriculum

Level 6.2 initializes from the passed Level 6.1 checkpoint, randomizes the
marker position in chunk 1, and expands through 2, 4, 8 and 16 chunks of 128
tokens. A shared probe supervises retention after every chunk.

```powershell
python run_level6_2_local.py
```

Each stage requires query, first probe and final probe accuracy >=95%, minimum
intermediate probe accuracy >=90%, twice consecutively.
