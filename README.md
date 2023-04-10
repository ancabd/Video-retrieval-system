# Video-retrieval-system
* To create a database run process.py and specify the folder with the videos to be added to the database with the parameter "training_set".
* To see the localization on a video run display_video.py and specify the video using the "input_path" parameter.
* To run the full pipeline including the localization and the querying run video_query.py with the video address as a parameter. To specify the start and end of the video clip, use "s" and "e" parameters and to specify the feature to be used use the "f" parameter. Using "-f all" uses both sift and color histogram as features to query with.
