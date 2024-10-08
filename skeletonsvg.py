import os
import numpy as np
import svgwrite
import folder_paths
from PIL import Image

class SkeletonSVG:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    DEPRECATED (`bool`):
        Indicates whether the node is deprecated. Deprecated nodes are hidden by default in the UI, but remain
        functional in existing workflows that use them.
    EXPERIMENTAL (`bool`):
        Indicates whether the node is experimental. Experimental nodes are marked as such in the UI and may be subject to
        significant changes or removal in future versions. Use with caution in production workflows.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "optional": {
                "custom_output_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    # RETURN_TYPES = ("IMAGE",)
    RETURN_TYPES = ()
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "saveSVG"
    OUTPUT_NODE = True

    CATEGORY = "SkeletonSVG"

    # binary image thinning (skeletonization) in-place.
    # implements Zhang-Suen algorithm.
    # http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
    # @param im   the binary image
    def thinningZS(self, im):
      prev = np.zeros(im.shape,np.uint8);
      while True:
        im = self.thinningZSIteration(im,0);
        im = self.thinningZSIteration(im,1)
        diff = np.sum(np.abs(prev-im));
        if not diff:
          break
        prev = im
      return im

    # 1 pass of Zhang-Suen thinning 
    def thinningZSIteration(self, im, iter):
      marker = np.zeros(im.shape,np.uint8);
      for i in range(1,im.shape[0]-1):
        for j in range(1,im.shape[1]-1):
          p2 = im[(i-1),j]  ;
          p3 = im[(i-1),j+1];
          p4 = im[(i),j+1]  ;
          p5 = im[(i+1),j+1];
          p6 = im[(i+1),j]  ;
          p7 = im[(i+1),j-1];
          p8 = im[(i),j-1]  ;
          p9 = im[(i-1),j-1];
          A  = (p2 == 0 and p3) + (p3 == 0 and p4) + \
               (p4 == 0 and p5) + (p5 == 0 and p6) + \
               (p6 == 0 and p7) + (p7 == 0 and p8) + \
               (p8 == 0 and p9) + (p9 == 0 and p2);
          B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          m1 = (p2 * p4 * p6) if (iter == 0 ) else (p2 * p4 * p8);
          m2 = (p4 * p6 * p8) if (iter == 0 ) else (p2 * p6 * p8);

          if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
            marker[i,j] = 1;

      return np.bitwise_and(im,np.bitwise_not(marker))


    def thinningSkimage(self, im):
      from skimage.morphology import skeletonize
      return self.skeletonize(im).astype(np.uint8)

    def thinning(self, im):
      try:
        return self.thinningSkimage(im)
      except:
        return self.thinningZS(im)

    #check if a region has any white pixel
    def notEmpty(self, im, x, y, w, h):
      return np.sum(im) > 0


    # merge ith fragment of second chunk to first chunk
    # @param c0   fragments from first  chunk
    # @param c1   fragments from second chunk
    # @param i    index of the fragment in first chunk
    # @param sx   (x or y) coordinate of the seam
    # @param isv  is vertical, not horizontal?
    # @param mode 2-bit flag, 
    #             MSB = is matching the left (not right) end of the fragment from first  chunk
    #             LSB = is matching the right (not left) end of the fragment from second chunk
    # @return     matching successful?             
    # 
    def mergeImpl(self, c0, c1, i, sx, isv, mode):

      B0 = (mode >> 1 & 1)>0; # match c0 left
      B1 = (mode >> 0 & 1)>0; # match c1 left
      mj = -1;
      md = 4; # maximum offset to be regarded as continuous
      
      p1 = c1[i][0 if B1 else -1];
      
      if (abs(p1[isv]-sx)>0): # not on the seam, skip
        return False
      
      # find the best match
      for j in range(len(c0)):
        p0 = c0[j][0 if B0 else -1];
        if (abs(p0[isv]-sx)>1): # not on the seam, skip
          continue
        
        d = abs(p0[not isv] - p1[not isv]);
        if (d < md):
          mj = j;
          md = d;

      if (mj != -1): # best match is good enough, merge them
        if (B0 and B1):
          c0[mj] = list(reversed(c1[i])) + c0[mj]
        elif (not B0 and B1):
          c0[mj]+=c1[i]
        elif (B0 and not B1):
          c0[mj] = c1[i] + c0[mj]
        else:
          c0[mj] += list(reversed(c1[i]))
        
        c1.pop(i);
        return True;
      return False;

    HORIZONTAL = 1;
    VERTICAL = 2;

    # merge fragments from two chunks
    # @param c0   fragments from first  chunk
    # @param c1   fragments from second chunk
    # @param sx   (x or y) coordinate of the seam
    # @param dr   merge direction, HORIZONTAL or VERTICAL?
    # 
    def mergeFrags(self, c0, c1, sx, dr):
      HORIZONTAL = 1;
      VERTICAL = 2;
      for i in range(len(c1)-1,-1,-1):
        if (dr == HORIZONTAL):
          if (self.mergeImpl(c0,c1,i,sx,False,1)):continue;
          if (self.mergeImpl(c0,c1,i,sx,False,3)):continue;
          if (self.mergeImpl(c0,c1,i,sx,False,0)):continue;
          if (self.mergeImpl(c0,c1,i,sx,False,2)):continue;
        else:
          if (self.mergeImpl(c0,c1,i,sx,True,1)):continue;
          if (self.mergeImpl(c0,c1,i,sx,True,3)):continue;
          if (self.mergeImpl(c0,c1,i,sx,True,0)):continue;
          if (self.mergeImpl(c0,c1,i,sx,True,2)):continue;      
        
      c0 += c1


    # recursive bottom: turn chunk into polyline fragments;
    # look around on 4 edges of the chunk, and identify the "outgoing" pixels;
    # add segments connecting these pixels to center of chunk;
    # apply heuristics to adjust center of chunk
    # 
    # @param im   the bitmap image
    # @param x    left of   chunk
    # @param y    top of    chunk
    # @param w    width of  chunk
    # @param h    height of chunk
    # @return     the polyline fragments
    # 
    def chunkToFrags(self, im, x, y, w, h):
      frags = []
      on = False; # to deal with strokes thicker than 1px
      li=-1; lj=-1;
      
      # walk around the edge clockwise
      for k in range(h+h+w+w-4):
        i=0; j=0;
        if (k < w):
          i = y+0; j = x+k;
        elif (k < w+h-1):
          i = y+k-w+1; j = x+w-1;
        elif (k < w+h+w-2):
          i = y+h-1; j = x+w-(k-w-h+3); 
        else:
          i = y+h-(k-w-h-w+4); j = x+0;
        
        if (im[i,j]): # found an outgoing pixel
          if (not on):     # left side of stroke
            on = True;
            frags.append([[j,i],[x+w//2,y+h//2]])
        else:
          if (on):# right side of stroke, average to get center of stroke
            frags[-1][0][0]= (frags[-1][0][0]+lj)//2;
            frags[-1][0][1]= (frags[-1][0][1]+li)//2;
            on = False;
        li = i;
        lj = j;
      
      if (len(frags) == 2): # probably just a line, connect them
        f = [frags[0][0],frags[1][0]];
        frags.pop(0);
        frags.pop(0);
        frags.append(f);
      elif (len(frags) > 2): # it's a crossroad, guess the intersection
        ms = 0;
        mi = -1;
        mj = -1;
        # use convolution to find brightest blob
        for i in range(y+1,y+h-1):
          for j in range(x+1,x+w-1):
            s = \
              (im[i-1,j-1]) + (im[i-1,j]) +(im[i-1,j+1])+\
              (im[i,j-1]  ) +   (im[i,j]) +    (im[i,j+1])+\
              (im[i+1,j-1]) + (im[i+1,j]) +  (im[i+1,j+1]);
            if (s > ms):
              mi = i;
              mj = j;
              ms = s;
            elif (s == ms and abs(j-(x+w//2))+abs(i-(y+h//2)) < abs(mj-(x+w//2))+abs(mi-(y+h//2))):
              mi = i;
              mj = j;
              ms = s;

        if (mi != -1):
          for i in range(len(frags)):
            frags[i][1]=[mj,mi]
      return frags;


    # Trace skeleton from thinning result.
    # Algorithm:
    # 1. if chunk size is small enough, reach recursive bottom and turn it into segments
    # 2. attempt to split the chunk into 2 smaller chunks, either horizontall or vertically;
    #    find the best "seam" to carve along, and avoid possible degenerate cases
    # 3. recurse on each chunk, and merge their segments
    # 
    # @param im      the bitmap image
    # @param x       left of   chunk
    # @param y       top of    chunk
    # @param w       width of  chunk
    # @param h       height of chunk
    # @param csize   chunk size
    # @param maxIter maximum number of iterations
    # @param rects   if not null, will be populated with chunk bounding boxes (e.g. for visualization)
    # @return        an array of polylines
    # 
    def traceSkeleton(self, im, x, y, w, h, csize, maxIter, rects):
      
      frags = []
      HORIZONTAL = 1
      VERTICAL = 2
      
      if (maxIter == 0): # gameover
        return frags;
      if (w <= csize and h <= csize): # recursive bottom
        frags += self.chunkToFrags(im,x,y,w,h);
        return frags;
      
      ms = im.shape[0]+im.shape[1]; # number of white pixels on the seam, less the better
      mi = -1; # horizontal seam candidate
      mj = -1; # vertical   seam candidate
      
      if (h > csize): # try splitting top and bottom
        for i in range(y+3,y+h-3):
          if (im[i,x]  or im[(i-1),x]  or im[i,x+w-1]  or im[(i-1),x+w-1]):
            continue
          
          s = 0;
          for j in range(x,x+w):
            s += im[i,j];
            s += im[(i-1),j];
          
          if (s < ms):
            ms = s; mi = i;
          elif (s == ms  and  abs(i-(y+h//2))<abs(mi-(y+h//2))):
            # if there is a draw (very common), we want the seam to be near the middle
            # to balance the divide and conquer tree
            ms = s; mi = i;
      
      if (w > csize): # same as above, try splitting left and right
        for j in range(x+3,x+w-2):
          if (im[y,j] or im[(y+h-1),j] or im[y,j-1] or im[(y+h-1),j-1]):
            continue
          
          s = 0;
          for i in range(y,y+h):
            s += im[i,j];
            s += im[i,j-1];
          if (s < ms):
            ms = s;
            mi = -1; # horizontal seam is defeated
            mj = j;
          elif (s == ms  and  abs(j-(x+w//2))<abs(mj-(x+w//2))):
            ms = s;
            mi = -1;
            mj = j;

      nf = []; # new fragments
      if (h > csize  and  mi != -1): # split top and bottom
        L = [x,y,w,mi-y];    # new chunk bounding boxes
        R = [x,mi,w,y+h-mi];
        
        if (self.notEmpty(im,L[0],L[1],L[2],L[3])): # if there are no white pixels, don't waste time
          if(rects!=None):rects.append(L);
          nf += self.traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects) # recurse
        
        if (self.notEmpty(im,R[0],R[1],R[2],R[3])):
          if(rects!=None):rects.append(R);
          self.mergeFrags(nf,self.traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mi,VERTICAL);
        
      elif (w > csize  and  mj != -1): # split left and right
        L = [x,y,mj-x,h];
        R = [mj,y,x+w-mj,h];
        if (self.notEmpty(im,L[0],L[1],L[2],L[3])):
          if(rects!=None):rects.append(L);
          nf+=self.traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects);
        
        if (self.notEmpty(im,R[0],R[1],R[2],R[3])):
          if(rects!=None):rects.append(R);
          HORIZONTAL = 1
          # self.mergeFrags(nf,self.traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mj,HORIZONTAL);
          self.mergeFrags(nf,self.traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mj,1);
        
      frags+=nf;
      if (mi == -1  and  mj == -1): # splitting failed! do the recursive bottom instead
        frags += self.chunkToFrags(im,x,y,w,h);
      
      return frags

    def create_svg_with_multiple_polylines(self, polylines, filename="output.svg", size=(500, 500)):
        """
        Create an SVG file with multiple polylines, all in black.
        
        :param polylines: List of polylines, where each polyline is a list of [x, y] coordinate pairs
        :param filename: Name of the output SVG file
        :param size: Tuple of (width, height) for the SVG canvas
        """
        dwg = svgwrite.Drawing(filename, size=size)
        
        # Create polylines and add them to the drawing
        for polyline_points in polylines:
            polyline = dwg.polyline(points=polyline_points, fill="none", stroke="black", stroke_width=2)
            dwg.add(polyline)
        
        # Save the drawing
        dwg.save()

    # def check_lazy_status(self, image, string_field, int_field, float_field, print_to_screen):
    # def check_lazy_status(self, image):
    #     """
    #         Return a list of input names that need to be evaluated.

    #         This function will be called if there are any lazy inputs which have not yet been
    #         evaluated. As long as you return at least one field which has not yet been evaluated
    #         (and more exist), this function will be called again once the value of the requested
    #         field is available.

    #         Any evaluated inputs will be passed as arguments to this function. Any unevaluated
    #         inputs will have the value None.
    #     """
    #     if(image):
    #       print(image)
    #     # if print_to_screen == "enable":
    #     #     return ["int_field", "float_field", "string_field"]
    #     # else:
    #     #     return []

    def generate_unique_filename(self, prefix, timestamp=False):
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d%H%M%S")
            return f"{prefix}_{timestamp_str}.svg"
        else:
            return f"{prefix}.svg"

    def saveSVG(self, images, filename_prefix="ComfyUI_SVG", append_timestamp=False, custom_output_path=""):
        output_path = custom_output_path if custom_output_path else self.output_dir
        os.makedirs(output_path, exist_ok=True)

        filename_prefix += self.prefix_append
        # full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, 1024, 1024)
        results = list()

        for (batch_number, img) in enumerate(images):
            unique_filename = self.generate_unique_filename(f"{filename_prefix}_{batch_number}", timestamp=True)
            final_filepath = os.path.join(output_path, unique_filename)
            print(final_filepath)

            i = 255. * img.cpu().numpy()
            i2 = np.clip(i, 0, 255).astype(np.uint8)

            im = (i2[:,:,0]>128).astype(np.uint8)
            im = self.thinning(im)
            rects = []
            polys = self.traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)

            # filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            # file = f"{filename_with_batch_num}_{counter:05}_.svg"
            self.create_svg_with_multiple_polylines(polys, final_filepath, (i.shape[1],i.shape[0]))

            results.append({
                "saved_svg": unique_filename, 
                "path": final_filepath
            })
            

        return { "ui": { "images": results } }






# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SkeletonSVG": SkeletonSVG
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SkeletonSVG": "SkeletonSVG"
}
