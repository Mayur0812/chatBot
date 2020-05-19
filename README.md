This is an on-prem implementation of a very basic chatBot using deep learning.
Idea is to create context aware embeddings of multiple sentences using pre-trained models.

* To create embeddings we have used tensorflow hub module provided by google
* From this module we have used universal sentence encoder to encode the questions/statements.
* Google's USE encodes each sentence into 512 dimensional vector keeping context awareness.
* You can read more about it here on it's official site:
  https://tfhub.dev/google/universal-sentence-encoder/1
* We have one-one mapping of questions to their answer (which could be extended to any cardinality as per business needs)
* Once encoding is done, we create K- Dimensional trtee of those embeddings.
* After encoding user query, tree based retrival of most similar sentence makes process faster rathen than iterting through all embeddings and finding max_K(all similarity scores)
