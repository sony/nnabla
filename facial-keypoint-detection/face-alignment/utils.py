# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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


def visualize(landmarks, image, output_path, plot=False):
    """Visualize the detected landmarks.

    Given a set of landmarks and image, plot the landmarks on the image and save it.

    Arguments:
        landmarks {list} -- list of the detected landmarks.
        image {numpy.array} -- an rgb image.
        output_path {string} -- path to save the visualized image.

    Keyword Arguments:
        plot {bool} -- define whether to plot or save.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    for i in range(len(landmarks)):
        pts_img = landmarks[i]
        ind = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        for i in range(len(ind) - 1):
            ax.plot(pts_img[ind[i]:ind[i + 1], 0], pts_img[ind[i]:ind[i + 1], 1], marker='o',
                    markersize=1, linestyle='-', color='w', lw=.75)
    plt.savefig(output_path)
    if plot:
        plt.show()
    plt.close(fig)
