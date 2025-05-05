# %%
from transformers import pipeline


# %%
summarizer = pipeline("summarization", model= "t5-base", tokenizer= "t5-base", framework="pt")

# %%
text = """Globalization is a multifaceted process characterized by the increasing interconnectedness and interdependence of countries worldwide. This integration spans economic, social, cultural, political, and technological dimensions, driven by factors such as advancements in communication and transportation, the liberalization of trade and capital flows, and the spread of information. Essentially, globalization signifies a shift towards a more integrated global society where borders become less significant in various aspects of life.   

Economically, globalization manifests in the expansion of international trade, the rise of multinational corporations, and the interconnectedness of financial markets. This has led to increased competition, the specialization of production across different nations, and the potential for greater economic growth. Consumers often benefit from a wider variety of goods and services at potentially lower prices. However, this economic integration also presents challenges such as increased competition for domestic industries, potential job displacement in some sectors, and the risk of global economic crises spreading rapidly."""

# %%
summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
print(summary[0]['summary_text'])


