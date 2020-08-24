# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from IPython.display import Image, display, Javascript
from google.colab.output import eval_js
from base64 import b64decode


def take_photo(filename="photo.png", cam_width=None, cam_height=None):
    """
        take a photo using a camera. If the machine has multiple cameras,
        you need to switch the camera
        --- Usage ---
        from IPython.display import Image
        try:
            filename = take_photo(cam_width=256, cam_height=256) # better to pass the default resolutions
            print('Saved to {}'.format(filename))
            # Show the image which was just taken.
            display(Image(filename))
        except Exception as err:
            # Errors will be thrown if the user does not have a webcam or if they do not
            # grant the page permission to access it.
            print(str(err))
    """
    js = Javascript('''
    async function takePhoto(cam_width, cam_height) {
      // default config
      var CONSTRAINTS = {
        audio: false,
        video: {
          width: {ideal: cam_width},
          height: {ideal: cam_height},
          deviceId: null
        }
      };

      function syncCamera(video, dev_id){
        CONSTRAINTS.video.deviceId = dev_id;
        if(curSTREAM !== null){
          curSTREAM.getVideoTracks().forEach((camera) => {
            camera.stop();
          });
      }

      navigator.mediaDevices.getUserMedia(CONSTRAINTS)
        .then( (stream) => {
          curSTREAM = stream;
          video.srcObject = stream;
          video.onloadedmetadata = (e) => {
            video.play();
          };
        })
      .catch((err) => {
        //console.log(err.name + ": " + err.message);
        return;
        });
      }

      // current Stream
      var curSTREAM = null;

      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        //console.log("enumerateDevices() not supported.");
        return;
      }

      navigator.mediaDevices.enumerateDevices()
      .then(function(devices) {
        devices.forEach(function(device) {
          if (device.kind.includes("video")) {
            const dev_button = document.createElement('button');
            dev_button.textContent = device.label;
            const dev_id = device.deviceId;
            div.appendChild(dev_button);
            dev_button.onclick = function() {
            syncCamera(video, dev_id);
            }
          };
        });
      })
      .catch(function(err) {
        //console.log(err.name + ": " + err.message);
        return;
      });

      const video = document.createElement('video');
      video.style.display = 'block';
      curSTREAM = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = curSTREAM;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      curSTREAM.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL();
      }
    ''')

    if not cam_width:
        cam_width = "null"
    if not cam_height:
        cam_height = "null"

    display(js)
    data = eval_js(f'takePhoto({cam_width}, {cam_height})')
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename
