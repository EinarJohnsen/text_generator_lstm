It is commonly known that Recurrent Neural Networks (RNN) in theory canmaintain context over a longer period of time, but this has shown to bedifficult in practice.  This project aims to expand upon the traditionalimplementations of Long Short-Term Memory (LSTM) RNN in order toachieve a better contextual understanding over a longer period of time,compared to existing implementations.  In this report we implement severalprototypes combining an LSTM with other types of networks, with variousminor alterations and for different use cases.  The results show that increasedperformance can be gained from this approach, especially on classificationtasks, but that the inherent nature of text generation makes it hard tomeasure the performance of different networks against each others.

Created by: Einar Johnsen, Christopher Dambakk and Jonas Heer

Data is taken from: 

Project Gutenberg’s Alice’s Adventures in Wonderland, by Lewis Carroll

This eBook is for the use of anyone anywhere at no cost and with
almost no restrictions whatsoever.  You may copy it, give it away or
re-use it under the terms of the Project Gutenberg License included
with this eBook or online at www.gutenberg.org


Title: Alice’s Adventures in Wonderland

Author: Lewis Carroll

Posting Date: June 25, 2008 [EBook #11]
Release Date: March, 1994
Last Updated: October 6, 2016

Language: English

Character set encoding: UTF-8

Initial code structure is based on code from: 

@misc{bhaveshoswal,
  author = {Bhavesh Vinod Oswal},
  title = {CNN-text-classification-keras},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bhaveshoswal/CNN-text-classification-keras}},
}

Jason Brownlee:
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


