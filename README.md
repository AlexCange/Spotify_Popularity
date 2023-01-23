# Spotify_Popularity

## SUMMARY / PROJECT SCOPE

Finishing our bootcamp in Data Science at WBS Coding school, my Partner Pat and I decided to work together for our final project. Being both music fans, he as a musician, and I as just fan of listening music, we had this idea to work with Spotify API. 
This tool is really easy to use, and full of information. 
The main goal of this project is to use our knowledge, internet, and Spotify API in order to predict if a track popularity can be predictable. In this case, it can guide artists to make some changes and to be aware of what can be expected while releasing a song. 

The Popularity on Spotify is a number between 0 to 100 where 100 is the most popular possible. 

> “The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.
Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past.” - Spotify Documentation

Having done some research on the internet, we could see similar projects with the Audio Features provided by the Spotify API. Also, mostly, the projects are classifying the tracks between Popular/Not Popular. We wanted then to expend this and have a real number output, and work also with the actual songs, not only the audio features. We could download 30secondes previews from spotify thanks to their API, and create spectrograms from it. 


## TOOLS / PROCESS USED

We worked locally at the beginning of the project, but due to high volume of data and processing, we needed to switch to Colab. We have used Tensorflow to build our model. 
We have decided to work on a concatenated model, with one pipe for the audio features, and the other pipe for the Spectrogram, then a last stage with all of these. 

After few days and due to short time, we decided to build first a model that takes only the audio features, in order to see if it would perform correctly. We could reach an absolute difference of **18 points on average (on 100, for a start, not too bad!).**

With the model taking also the Spectrograms, we could reach an **absolute difference of 13 points in average**.

## CONCLUSION

We have learned a lot in these 3 weeks, and we are pretty happy of the results we got. We have worked with the playlist Friday Discovery and we have found the archives of the playlist on Github. We thought it was a good playlist, with a large variety of artists (known, and unknown) and also different genres. 

We got limited anyway since we started with 16,000 songs, and ended up with 10,000 songs due to data cleaning, and processing the MP3 to WAV to WAVEFORM to SPECTROGRAMS. Also, our model does not take in consideration the released date. It works better on recent music than old songs. 

To answer the question ***Can the popularity of a track on Spotify be predicted?*** I guess the answer is yes, with limits of course. 

