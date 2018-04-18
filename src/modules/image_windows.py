import numpy as np
import math

def get_num_windows(image_height, image_width, window_height, window_width, overlap=True, overlap_corners=True):
    """Return the number of height and width windows to generate

    Args:
        image_height: Height of the image in pixels.
        image_width: Width of the image in pixels.
        window_height: The height of each generated window.
        window_width: The width of each generated window.
        overlap: True iff generated windows should be overlapped.
        overlap_corners: True iff the corners/edges should be mirrored and contain
          as many overlaps as the centre of the image.

    Returns:
        A 3-element list containing the total number of windows, and the number of
        windows needed for the image height and width, respectively.
    """
    if not overlap:
        num_height_windows = math.ceil(image_height/window_height)
        num_width_windows = math.ceil(image_width/window_width)
    elif overlap_corners:
        num_height_windows = math.ceil(image_height/(window_height/2)) + 1
        num_width_windows = math.ceil(image_width/(window_width/2)) + 1
    else:
        num_height_windows = max(1, math.ceil(image_height/(window_height/2)) - 1)
        num_width_windows = max(1, math.ceil(image_width/(window_width/2)) - 1)

    return ((num_height_windows*num_width_windows), num_height_windows, num_width_windows)


def split_image_to_windows(image, window_height, window_width, overlap=True, overlap_corners=True):
    """Split an input image into multiple equal-sized windows.
    
    The input image can have any dimensions. It will be split into multiple
    equal-sized windows.

    Args:
        image: Input image with 3 dimensions - height, width, and channels.
        window_height: The height of each generated window.
        window_width: The width of each generated window.
        overlap: True iff generated windows should be overlapped.
        overlap_corners: True iff the corners/edges should be mirrored and contain
          as many overlaps as the centre of the image.

    Returns:
        Numpy 4-d array, dimensions are the window index, window height, window
        width, channel data. The returned shape[0] describes how many total
        windows were created.
    """
    img_height = image.shape[0]
    img_width = image.shape[1]

    (total_windows, num_height_windows, num_width_windows) = get_num_windows(img_height, img_width,
                                                                             window_height, window_width,
                                                                             overlap, overlap_corners)

    window_images = np.zeros((total_windows, window_height, window_width, image.shape[2]), image.dtype)

    src_window_height = int(window_height/2)
    src_window_width = int(window_width/2)

    # Training with overlapping windows, including overlaps for corners/edges, provides the
    # best IoU/LB scores. If training was not done using overlaps for corners/edges, then
    # predictions using overlaps without additional corner/edge windows provided the best
    # IoU/LB score.
    if not overlap_corners:
        src_window_height = window_height
        src_window_width = window_width
        if overlap > 0:
            src_window_height = int(src_window_height/2)
            src_window_width = int(src_window_width/2)

        cur_window = 0
        for h in range(num_height_windows):
            for w in range(num_width_windows):
                # The right-most and bottom-most windows of the image may need to
                # be padded. If we are at the edge of the input image, we will pad
                # the output windows. Determine how much is left to copy from the
                # input image, and the padding will then be the difference between
                # the end of the window, and the edge of the input image.
                image_edge_height = min(img_height, window_height + (h * src_window_height))
                image_edge_width = min(img_width, window_width + (w * src_window_width))

                window_pad_height = (window_height + (h * src_window_height)) - image_edge_height
                window_pad_width = (window_width + (w * src_window_width)) - image_edge_width

                window_images[cur_window] = np.pad(image[h * src_window_height:image_edge_height,
                                                         w * src_window_width:image_edge_width,:],
                                                   ((0,window_pad_height),(0,window_pad_width),(0,0)),
                                                   'symmetric')
                cur_window += 1
    else:
        cur_window = 0
        for h in range(num_height_windows):
            # The corners and edges need to be padded using symmetric padding.
            src_top = max(0, (h - 1) * src_window_height)
            src_bottom = min(img_height, (h + 1) * src_window_height)

            pad_top = 0
            if h == 0:
                pad_top = src_window_height
            pad_bottom = window_height - pad_top - (src_bottom - src_top)

            for w in range(num_width_windows):
                src_left = max(0, (w - 1) * src_window_width)
                src_right = min(img_width, (w + 1) * src_window_width)

                pad_left = 0
                if w == 0:
                    pad_left = src_window_width
                pad_right = window_width - pad_left - (src_right - src_left)

                window_images[cur_window] = np.pad(image[src_top:src_bottom,src_left:src_right,:],
                                                   ((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),
                                                   'symmetric')
                cur_window += 1

    return window_images

def stitch_single_image(windows, image_height, image_width, overlap=True, overlap_corners=True):
    """Stitch together an image that had been split into multiple equal-sided windows.

    Args:
        windows: All the windows of the image to be stitched together.
        image_height: The original height of the image to be stitched back together.
        image_width: The original width of the image to be stitched back together.
        overlap: True iff generated windows should be overlapped.
        overlap_corners: True iff the corners/edges should be mirrored and contain
          as many overlaps as the centre of the image.

    Returns:
        Numpy 3-d array/image of the specified image_height and image_width.
    """
    window_height = windows[0].shape[0]
    window_width = windows[0].shape[1]
    (_, num_height_windows, num_width_windows) = get_num_windows(image_height, image_width,
                                                                 window_height, window_width,
                                                                 overlap, overlap_corners)

    dest_img = np.zeros((image_height, image_width, windows[0].shape[2]), dtype=windows[0].dtype)
    cur_window = 0

    # Training with overlapping windows, including overlaps for corners/edges, provides the
    # best IoU/LB scores. If training was not done using overlaps for corners/edges, then
    # predictions using overlaps without additional corner/edge windows provided the best
    # IoU/LB score.
    dest_window_height = window_height
    dest_window_width = window_width
    if not overlap:
        # No overlapping windows
        for h in range(num_height_windows):
            # dest is full-size img, src is current window
            end_start_height = h * window_height
            end_dest_height = min(image_height, end_start_height + window_height)
            end_src_height = end_dest_height - end_start_height

            for w in range(num_width_windows):
                end_start_width = w * window_width
                end_dest_width = min(image_width, end_start_width + window_width)
                end_src_width = end_dest_width - end_start_width
                src_image = windows[cur_window]
                cur_window += 1
                dest_img[end_start_height:end_dest_height,end_start_width:end_dest_width] = src_image[0:end_src_height,0:end_src_width]
    elif overlap_corners:
        dest_window_height = int(dest_window_height/2)
        dest_window_width = int(dest_window_width/2)

        for h in range(num_height_windows):
            # dest is full-size img, src is current window
            dest_top = max(0, (h - 1) * dest_window_height)
            dest_bottom = min(image_height, (h + 1) * dest_window_height)

            if h == 0:
                src_top = dest_window_height
            else:
                src_top = 0
            src_bottom = src_top + (dest_bottom - dest_top)

            for w in range(num_width_windows):
                dest_left = max(0, (w - 1) * dest_window_width)
                dest_right = min(image_width, (w + 1) * dest_window_width)
                if w == 0:
                    src_left = dest_window_width
                else:
                    src_left = 0
                src_right = src_left + (dest_right - dest_left)

                # Scale down the input window, the edges won't have 4 overlapping windows,
                # (they will have either 1 or 2 overlapping windows), but we discard all
                # edges anyways.
                src_image = windows[cur_window].copy()
                src_image[:,:] = src_image[:,:] / 4

                cur_window += 1
                dest_img[dest_top:dest_bottom,dest_left:dest_right] += src_image[src_top:src_bottom,src_left:src_right]
    else:
        if image_height > dest_window_height:
            dest_window_height = int(dest_window_height/2)
        if image_width > dest_window_width:
            dest_window_width = int(dest_window_width/2)
 
        for h in range(num_height_windows):
             # dest is full-size img, src is current window
            dest_start_height = h * dest_window_height
            dest_end_height = min(image_height, dest_start_height + window_height)
            src_end_height = dest_end_height - dest_start_height
 
            for w in range(num_width_windows):
                dest_start_width = w * dest_window_width
                dest_end_width = min(image_width, dest_start_width + window_width)
                src_end_width = dest_end_width - dest_start_width
 
                src_image = windows[cur_window].copy()
                # Scale down the input window based on how many windows overlap each
                # quarter of the window.
                q1 = 1
                q2 = 1
                q3 = 1
                q4 = 1
                if (h == 0 and w > 0) or (w == 0 and h > 0):
                    q1 = 2
                elif (h > 0 and w > 0):
                    q1 = 4
                if (h == 0 and w < (num_width_windows - 1)) or (h > 0 and w == (num_width_windows - 1)):
                    q2 = 2
                elif (h > 0 and w < (num_width_windows - 1)):
                    q2 = 4
                if (h < (num_height_windows - 1) and w == 0) or (h == (num_height_windows - 1) and w > 0):
                    q3 = 2
                elif (h < (num_height_windows - 1)) and w > 0:
                    q3 = 4
                if (h < (num_height_windows - 1) and w == (num_width_windows - 1)) or (h == (num_height_windows - 1) and w < (num_width_windows - 1)):
                    q4 = 2
                elif (h < (num_height_windows - 1)) and (w < (num_width_windows - 1)):
                    q4 = 4
                src_image[0:dest_window_height,0:dest_window_width] = src_image[0:dest_window_height,0:dest_window_width] / q1
                src_image[0:dest_window_height,dest_window_width:window_width] = src_image[0:dest_window_height,dest_window_width:window_width] / q2
                src_image[dest_window_height:window_height,0:dest_window_width] = src_image[dest_window_height:window_height,0:dest_window_width] / q3
                src_image[dest_window_height:window_height,dest_window_width:window_width] = src_image[dest_window_height:window_height,dest_window_width:window_width] / q4
 
                cur_window += 1
                dest_img[dest_start_height:dest_end_height,dest_start_width:dest_end_width] += src_image[0:src_end_height,0:src_end_width]


    return dest_img

def stitch_all_images(windows, sizes, overlap=True, overlap_corners=True):
    """Stitch together the provided windows into a list of original-sized images.

    Args:
        windows: All the windows of all the images to be stitched together.
        sizes: The original image sizes.
        overlap: True iff generated windows should be overlapped.
        overlap_corners: True iff the corners/edges should be mirrored and contain
          as many overlaps as the centre of the image.

    Returns:
        List of numpy 3-d array/images stitched together from the input windows.
    """
    stitched = []
    cur_window = 0
    window_height = windows[0].shape[0]
    window_width = windows[0].shape[1]

    for (image_height, image_width) in sizes:
        (image_windows, _, _) = get_num_windows(image_height, image_width, window_height, window_width, overlap, overlap_corners)
        end_window = cur_window + image_windows
        stitched.append(stitch_single_image(windows[cur_window:end_window], image_height, image_width, overlap, overlap_corners))
        cur_window = end_window

    return stitched
