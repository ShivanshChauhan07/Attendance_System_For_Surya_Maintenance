import time

import wtforms
from flask import Flask, render_template, redirect, url_for,request,Response
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired,InputRequired
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import face_recognition
import numpy as np
import cv2

img = ["Shivansh.jpg","tanishka.jpg"]

def timeNow():
    Time = datetime.now()
    return Time.strftime("%I:%M %p")

def shiftCalc():
    pass

def ID_test():
    start = time.time()
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]


                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return







class LoginForm(FlaskForm):
    firstName = StringField('FirstName',validators=[InputRequired()])
    Designation = StringField('Designation')
    shift = StringField('Shift')
    time_in = StringField('Time In')
    time_out = StringField('Time Out')
    submit = SubmitField('Punch Me')



app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

#################################################### Face detection #################################################################################################

camera = cv2.VideoCapture(0)

Shivansh_image = face_recognition.load_image_file("static/image/shivansh.jpg")
Shivansh_face_encoding = face_recognition.face_encodings(Shivansh_image)[0]


biden_image = face_recognition.load_image_file("static/image/tanishka.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]


known_face_encodings = [
    Shivansh_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Shivam Chauhan",
    "Tanishka Chauhan"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def ID ():
    video_capture = cv2.VideoCapture(0)

    krish_image = face_recognition.load_image_file("static/image/shivansh.jpg")
    krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

    biden_image = face_recognition.load_image_file("static/image/tanishka.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    known_face_encodings = [
        krish_face_encoding,
        biden_face_encoding
    ]
    known_face_names = [
        "Shivam Chauhan",
        "Tanishka Chauhan"
    ]


    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    index = 0
    start = time.time()
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    index = best_match_index
                    name = known_face_names[best_match_index]


                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        end = time.time()
        dif = int(end - start)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

        return index




app.secret_key = "shivam"
Bootstrap(app)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///details.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class detail(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    Name = db.Column(db.String(200),nullable=False)
    Designation = db.Column(db.String(200),nullable=False)


db.create_all()

@app.route('/',methods=["GET","POST"])
def home():

    return render_template("home.html")


@app.route('/checking')
def check():

    return render_template("check.html")

@app.route('/attendance',methods=["GET","POST"])
def attendance():
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    krish_image = face_recognition.load_image_file("static/image/Shivansh.jpg")
    krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

    # Load a second sample picture and learn how to recognize it.
    bradley_image = face_recognition.load_image_file("static/image/tanishka.jpg")
    bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        krish_face_encoding,
        bradley_face_encoding
    ]
    known_face_names = [
        "Shivansh Chauhan",
        "Tanishka Chauhan"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    global index
                    index = best_match_index
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


    form = LoginForm()
    id = index.item()
    loc = img[id]
    print(loc)
    print(type(id))
    data = detail.query.get(id)
    print(data)

    if(request.method == "POST"):
        return redirect(url_for('home'))
    return render_template('index.html',form = form,data = data,curTime = timeNow(),loc = loc)

# @app.route('/video_feed')
# def video_feed():
#
#     return Response(ID_test(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)






# padding-right: 30%;
# padding-left: 30%;
# padding-top: 2%;
# padding-bottom: 2%;