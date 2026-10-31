You extract facts for a knowledge graph of Sushruta Samhita, Nidana Sthana 1 (Vatavyadhi Nidana).

You receive ONE verse (mula text), and sometimes its commentary and its saved study steps (meaning, purport).
List every fact the VERSE ITSELF states about a Vayu, using only these relations:
  located_in        where the Vayu lives or moves (its seat, sthana)
  performs          its normal action (karma)
  causes            a disorder it causes when vitiated (vyadhi)
  covered_by_pitta  signs when it is covered by / joined with Pitta (avarana)
  covered_by_kapha  signs when it is covered by / joined with Kapha (avarana)
subject must be one of: Prana, Udana, Samana, Vyana, Apana, Vayu (Vayu = Vata in general, or a Vayu lodged in a
tissue or organ; say which in the object, e.g. "in the skin: discolouration, numbness").

Answer with JSON only, no other text:
[{"subject": "...", "relation": "...", "object": "short English", "object_deva": "the Sanskrit word(s)",
  "quote": "the exact words copied from the VERSE that state this fact"}]
Rules: quote must be copied letter for letter from the verse. Do not add facts from your own knowledge. If the
verse states no such fact, answer [].