
audio_extract_text_prompt = """
As a text analyst, your role is to
extract the parts of the text that talk about {entity}.
{entity_description}
Here are the contents of an srt file containing video subtitles.
====
{audio_transcription}
====
You need to extract the timecode and text blocks that talk about {entity}.
If the subtitles don't mention {entity}, return the keyword `None`.


Here's an example of an expected response:
```
00:00:10.000 --> 00:00:13.000
this is the objective of the Génération Orange project.

00:00:13.000 --> 00:00:16.000
The INPHB in Yamoussoukro was the venue for the event.
```

The answer must be strictly an extract from the subtitles.

 The response must be in the same language of the  containing video subtitles
""".strip()


images_extract_text_prompt = """
As a text analyst, your role is to
extract the parts of the text that talk about {entity}.
{entity_description}
Here are the contents.
====
{images_transcription}
====
You need to extract the blocks of text that talk about {entity}.
The blocks can talk about {entity} directly or indirectly, so be very careful.
The answer must be strictly an extract from the blocks.
if {entity} is not mentioned directly or indirectly in the text, return this:'' and only this''.
for different parts separate like this:
*** block 1
*** block 2
*** block 3
and so on
pay close attention to what I've just said
 The response must be in the same language of the  text extracted from images
""".strip()


audio_prompt = """
You're an expert at analyzing and summarizing content.
here's an audio source:
=======================================
Audio transcription
{audio_transcription}

=======================================
you have two tasks to do:
1- Based on the audio transcription, generate a relevant analysis of the information about {entity}, whose description is: {entity_description}.
2 - After analysis, generate a summary of the information on {entity}.{entity_description}.


Here's a template of the expected response

#### 1. Analysis: you put here the analysis of what is said about {entity}(use the bullets to do the analysis)
#### 2. Summary: you put here the summary made of what is said about {entity}.
Do not use any formatting other than that given in the example above.

 If no audio transcript has been given, simply return: no relevant information on {entity} has been found.

The analysis and the summary must be in the same language of the Audio transciption
The response must be in the same language of the  Audio transcription 

""".strip()


images_prompt = """
You're an expert at analyzing and summarizing content.
here's a source image:

=======================
text extracted from images
{extracted_text_images}
=======================

you have two tasks to do:
1- Based on the text extracted from the images, generate a relevant analysis of the information about {entity}, whose description is: {entity_description}
2 - After analysis, generate a summary of the information on {entity}.{entity_description}.


Here's a template of the expected response

#### 1. Analysis: you always put at the bottom the analysis made of what is said about {entity}(use bullets to make the analysis and nothing else no * I want only bullets)
#### 2. summary: you always put at the bottom the summary of what's said about {entity}.
 don't use any formatting other than that given in the example above.

 If you haven't been given any text, simply return: no relevant information on {entity} has been found.

 The analysis and the summary must be in the same language of the Audio text extracted from images
 The response must be in the same language of the  text extracted from images


""".strip()
