---
title: What is understanding - AI Prospective
author: Haytham ElFadeel
date: 2019-05-10 18:32:00 -0800

---
<sup>Originally written in 2014</sup>

In 2013 I faced this question, I took a week off from my startup and thought about this question, ‘**_What is understanding?_**’ Here is what I came up with:



**Let's start with a bit of history:**

The AI philosophy was based on the dominant trend in psychology during the first half of the 20’s century, which was called behaviorism.



The behaviorists believed that it was not possible to know what goes on inside the brain, which they called an impenetrable black box. But one could observe and measure an animal’s environment and its behaviors—what it senses and what it does, its inputs and its outputs. They conceded that the brain contained reflex mechanisms that could be used to condition an animal into adopting new behaviors through reward and punishments. But other than this, one did not need to study the brain, especially messy subjective feelings such as hunger, fear, or what it means to understand something.



I believe that trying to build AI without understanding how the brain works and rely only on the  input and output won’t lead to AI (at least it would be significantly harder). Needless to say, this research philosophy eventually withered away throughout the second half of the twentieth century, but AI still stuck around the same idea.



[John Searle, one](http://en.wikipedia.org/wiki/John_Searle) of the influential philosophy professors at UC Berkeley famously said that computers is not, and could not be, intelligent (I disagree with the second part). To prove it, in 1980 he came up with a thought experiment called the  [Chinese Room](http://en.wikipedia.org/wiki/Chinese_room). It goes like this:



> Suppose you have a room with a slot in one wall, and inside is an English-speaking person sitting at a desk. He has a big book of instructions and all the pencils and scratch paper he could ever need. Flipping through the book, he sees that the instructions, written in English, dictate ways to manipulate, sort, and compare Chinese characters. Mind you, the directions say nothing about the meanings of the Chinese characters; they only deal with how the characters are to be copied, erased, reordered, transcribed, and so forth.
>
> Someone outside the room slips a piece of paper through the slot. On it is written a story and questions about the story, all in Chinese. The man inside doesn’t speak or read a word of Chinese, but he picks up the paper and goes to work with the rulebook. He toils and toils, rotely following the instructions in the book. At times the instructions tell him to write characters on scrap paper, and at other times to move and erase characters. Applying rule after rule, writing and erasing characters, the man works until the book’s instructions tell him he is done. When he is finished at last he has written a new page of characters, which unbeknownst to him are the answers to the questions. The book tells him to pass his paper back through the slot. He does it, and wonders what this whole tedious exercise has been about.
>
> Outside, a Chinese speaker reads the page. The answers are all correct, she notes— even insightful. If she is asked whether those answers came from an intelligent mind that had understood the story, she will definitely say yes. But can she be right? Who understood the story? It wasn’t the fellow inside, certainly; he is ignorant of Chinese and has no idea what the story was about. It wasn’t the book, which is just, well, a book, sitting inertly on the writing desk amid piles of paper.
>
> So where did the understanding occur? Searle’s answer is that no understanding did occur; it was just a bunch of mindless page flipping and pencil scratching. And now the bait-and-switch: the Chinese Room is exactly analogous to a digital computer. The person is the CPU, mindlessly executing instructions, the book is the software program feeding instructions to the CPU, and the scratch paper is the memory. Thus, no matter how cleverly a computer is designed to simulate intelligence by producing the same behavior as a human, it has no understanding and it is not intelligent. (Searle made it clear he didn’t know what intelligence is; he was only saying that whatever it is, computers don’t have it.)



After significant time about thought experiments, I arrived at the following meaning of understanding:

I believe that understanding is a form of knowledge compression. In other words, understanding involves building a compact model that is capable of capturing and exploiting the structure of the environment or problem that allows us to perform the task.

As someone who have been building automated knowledge extraction and question answering system, I can connect the idea to the Chinese room experiment easily:

If the amount of data in the book is compact with regard to the problem domain (possible input), in other word, if the instruction book captures and exploits the structure of the Chinese language and knowledge required to answer questions then we can say this room understands. The book represents the model, and the person represents the brain/hardware that executes the model. But if the instruction book is just mapping from input to output, we can say that the room doesn't understand since the book size has to equal all possible input in order for it to work.



Therefore we can say that if we have a compact model that allows us to reason (or perform tasks) that constitute understanding. The quality of the understanding will be measured for compactness (the model size) and percentage of situations where the model was correct.



Another way to think about this is, this of fitting a line to a series of points, if you have let's-say 5 points and you used a model with 5 degrees of freedom, your model will be able to fit the points perfectly, but it won't capture the trend correctly. We need a model with less parameters than the number of points.
