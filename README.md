Subtitle Synchronization is an interesting problem to solve with the application of 
Machine Learning and Probability.

There are already quite a few python libraries that deal with this problem and this
library is inspired from a famous library called 'autosubsync' [https://github.com/oseiskar/autosubsync].

Subtitle Synchronization has two major processing steps:

--> Vocal section identification
    This processing is associated with identification of vocal section present in video file. This library
uses Spleeter, open source audio separation library for this purpose and then uses audio processing
technique to build the vocal classification model.

This library differs from 'autosubsync' on this step. 'autosubsync' uses the labeling of vocal section
with reference from SRT file. This has some inherent limitations with regard to purpose of SRT file.
We would like to address this limitation with the application of Spleeter and other vocal processing
techniques.

We achieve very high accuracy with the current method and we believe this improved accuracy
would have a positive impact on the overall accuracy of subtitle synchronization.

--> Subtitle correction
    Subtitle timing correction based on 'Vocal Classification' results is second and final process involved in this activity.
Here, we reuse the processing of 'autosubsync' library completely. Author of autosubsync has done an
outstanding job in applying 'bernoulli equation' in solving this step. It was a
very good learning for us to understand the application. 

Pls refer this blog for the detailed information about the processing involved in this library. 