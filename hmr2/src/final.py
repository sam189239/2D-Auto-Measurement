from utils import *

def process(ht):
  start_top = time.time()
  ## Parameters ##
  front_img_dir = "..\\..\\in\\front.jpg"
  side_img_dir = "..\\..\\in\\side.jpg"

  ht_factor = 5 # percentage of pixels to detect top of segmentation in the mask
  neck_shift_factor = 0.025 # shifting neck point upward from models predicted point
  waist_shift_factor = 0.04 # shifting hip point upward from models predicted point
  arm_scale = 0.95 
  ht_scale = 1.04
  wrist_factor = 2.2

  ## Front Image ##
  start_front = time.time()
  front_image = Image.open(front_img_dir)
  res_im,seg_f=seg_model.run(front_image)
  mask_f, bg_removed_f = bg_removal(front_image, seg_f)

  ori_f, inp_f, param_f = preprocess_image("..\\..\\out\\bg_rem_img.jpg",224)
  joints_f, vertices_f, cams_f = hmr(inp_f, hmr_model)
  joints_f = joints_f[0]
  img_f, joints_f = convert_to_int(inp_f, joints_f)

  mask_scaled_f = preprocess_gray(mask_f)
  top_f, bottom_f  = ht_pts(mask_scaled_f,ht_factor)
  ht_p_f = (joints_f[0][1]+joints_f[5][1])/2 - top_f
  shifted_neck_f = shift_neck(joints_f, neck_shift_factor, ht_p_f)
  mask_crop_f = mask_scaled_f[top_f:int((joints_f[8][1]+joints_f[9][1])/2),joints_f[8][0]:joints_f[9][0]] 

  edges_f = cv2.Canny(mask_scaled_f,224,224)
  dest_scaled_f = edges_f
  dest_crop_f = dest_scaled_f[joints_f[13][1]:int((joints_f[8][1]+joints_f[9][1])/2),joints_f[8][0]:joints_f[9][0]]
  norm_f = abs(dest_crop_f)>0
  r_neck_f,l_neck_f = shortest_neck(norm_f,[shifted_neck_f[0]-joints_f[8][0],shifted_neck_f[1]-joints_f[13][1]])
  r_neck_scaled_f = [r_neck_f[1]+joints_f[8][0],r_neck_f[0]+joints_f[13][1]]
  l_neck_scaled_f = [l_neck_f[1]+joints_f[8][0],l_neck_f[0]+joints_f[13][1]] 


  shifted_waist_f = shift_waist(joints_f, ht_p_f, waist_shift_factor)
  r_waist_f,l_waist_f = waist_pts(abs(edges_f)>0,shifted_waist_f)

  r_waist_f = [r_waist_f,shifted_waist_f[1]]
  l_waist_f = [l_waist_f,shifted_waist_f[1]]

  ## Hand Tracking - Edge Detection ##
  hand_crop_r = (edges_f)[int(joints_f[6][1]-ht_p_f * 0.15):int(joints_f[6][1]+ht_p_f * 0.15), int(joints_f[6][0]-ht_p_f * 0.15):int(joints_f[6][0]+ht_p_f * 0.15)]
  shifted_hand_r = np.array([ht_p_f * 0.15,ht_p_f * 0.15]) 
  r,l = shortest_hand_right(hand_crop_r, shifted_hand_r)
  right_wrist_r = np.array([r[1]+joints_f[6][0]-ht_p_f * 0.15,r[0]+joints_f[6][1]-ht_p_f * 0.15])
  right_wrist_l = np.array([l[1]+joints_f[6][0]-ht_p_f * 0.15,l[0]+joints_f[6][1]-ht_p_f * 0.15])

  hand_crop_l = (edges_f)[int(joints_f[11][1]-ht_p_f * 0.15):int(joints_f[11][1]+ht_p_f * 0.15), int(joints_f[11][0]-ht_p_f * 0.15):int(joints_f[11][0]+ht_p_f * 0.15)]
  shifted_hand_l = np.array([ht_p_f * 0.15,ht_p_f * 0.15]) 
  r,l = shortest_hand_left(hand_crop_l, shifted_hand_l)
  left_wrist_r = np.array([r[1]+joints_f[11][0]-ht_p_f * 0.15,r[0]+joints_f[11][1]-ht_p_f * 0.15])
  left_wrist_l = np.array([l[1]+joints_f[11][0]-ht_p_f * 0.15,l[0]+joints_f[11][1]-ht_p_f * 0.15])

  end_front = time.time()
  print("Processed Front Image... Time taken: " + str(end_front-start_front)+ " seconds.")


  ## Side Image ##
  start_side = time.time()
  side_image = Image.open(side_img_dir)
  res_im,seg_s=seg_model.run(side_image)
  mask_s, bg_removed_s = bg_removal(side_image, seg_s)

  ori_s, inp_s, param_s = preprocess_image("..\\..\\out\\bg_rem_img.jpg",224)
  joints_s, vertices_s, cams_s = hmr(inp_s, hmr_model)

  joints_s = joints_s[0]
  img_s, joints_s = convert_to_int(inp_s, joints_s)

  mask_scaled_s = preprocess_gray(mask_s)
  top_s, bottom_s  = ht_pts(mask_scaled_s,ht_factor)
  ht_p_s = (max(joints_s[0][1],joints_s[5][1])) - top_s
  shifted_neck_s = shift_neck(joints_s, neck_shift_factor, ht_p_s)
  mask_crop_s = mask_scaled_s[top_s:int((joints_s[8][1]+joints_s[9][1])/2),int(joints_s[8][0]-ht_p_s*0.05):int(joints_s[9][0]+ht_p_s*0.05)]

  ## Edge detection Method
  edges_s = cv2.Canny(mask_scaled_s,224,224)
  dest_scaled_s = edges_s
  dest_crop_s = dest_scaled_s[joints_s[13][1]:int((joints_s[8][1]+joints_s[9][1])/2),int(joints_s[8][0]-ht_p_s*0.1):int(joints_s[9][0]+ht_p_s*0.1)]
  norm_s = abs(dest_crop_s)>0
  r_neck_s,l_neck_s = shortest_neck(norm_s,[joints_s[12][0]-joints_s[8][0]+ht_p_s*0.1, joints_s[12][1]-ht_p_s*neck_shift_factor-joints_s[13][1]])

  r_neck_scaled_s = [r_neck_s[1]+joints_s[8][0]-ht_p_s*0.1,r_neck_s[0]+joints_s[13][1]]
  l_neck_scaled_s = [l_neck_s[1]+joints_s[8][0]-ht_p_s*0.1,l_neck_s[0]+joints_s[13][1]]


  shifted_waist_s = shift_waist(joints_s, ht_p_s, waist_shift_factor)
  # r_waist_s,l_waist_s = waist_pts(mask_scaled_s,shifted_waist_s)
  r_waist_s,l_waist_s = waist_pts(abs(edges_s)>0,shifted_waist_s)

  r_waist_s = [r_waist_s,shifted_waist_s[1]]
  l_waist_s = [l_waist_s,shifted_waist_s[1]]
  end_side = time.time()
  print("Processed Side Image...  Time taken: " + str(end_side-start_side)+ " seconds.")

  

  ## Feature points ##
  plt.subplot(221)
  plt.imshow(mask_scaled_f)
  plt.plot(r_waist_f[0],r_waist_f[1], marker='.', color="blue")
  plt.plot(l_waist_f[0],l_waist_f[1], marker='.', color="blue")
  plt.plot(r_neck_scaled_f[0],r_neck_scaled_f[1],marker = ".", color = 'blue')
  plt.plot(l_neck_scaled_f[0],l_neck_scaled_f[1],marker = ".", color = 'blue')
  plt.plot(joints_f[12][0], joints_f[12][1],marker = ".", color = 'red')
  plt.plot(left_wrist_r[0],left_wrist_r[1],marker='.',color = 'blue')
  plt.plot(left_wrist_l[0],left_wrist_l[1],marker='.',color = 'blue')
  plt.plot(right_wrist_r[0],right_wrist_r[1],marker='.',color = 'blue')
  plt.plot(right_wrist_l[0],right_wrist_l[1],marker='.',color = 'blue')
  for i in [2,3,6,11]:
    plt.plot(joints_f[i][0], joints_f[i][1], marker='.', color="red")

  plt.subplot(222)
  plt.imshow(mask_scaled_s)
  plt.plot(r_waist_s[0],r_waist_s[1], marker='.', color="blue")
  plt.plot(l_waist_s[0],l_waist_s[1], marker='.', color="blue")
  plt.plot(r_neck_scaled_s[0],r_neck_scaled_s[1],marker = ".", color = 'blue')
  plt.plot(l_neck_scaled_s[0],l_neck_scaled_s[1],marker = ".", color = 'blue')
  plt.plot(joints_s[12][0], joints_s[12][1],marker = ".", color = 'red')
  for i in [2,3]:
    plt.plot(joints_s[i][0], joints_s[i][1], marker='.', color="red")

  plt.subplot(223)
  plt.imshow((((inp_f / 2.)+0.5)*255).astype(int))
  plt.plot(r_waist_f[0],r_waist_f[1], marker='.', color="blue")
  plt.plot(l_waist_f[0],l_waist_f[1], marker='.', color="blue")
  plt.plot(r_neck_scaled_f[0],r_neck_scaled_f[1],marker = ".", color = 'blue')
  plt.plot(l_neck_scaled_f[0],l_neck_scaled_f[1],marker = ".", color = 'blue')
  plt.plot(joints_f[12][0], joints_f[12][1],marker = ".", color = 'red')
  plt.plot(left_wrist_r[0],left_wrist_r[1],marker='.',color = 'blue')
  plt.plot(left_wrist_l[0],left_wrist_l[1],marker='.',color = 'blue')
  plt.plot(right_wrist_r[0],right_wrist_r[1],marker='.',color = 'blue')
  plt.plot(right_wrist_l[0],right_wrist_l[1],marker='.',color = 'blue')
  for i in [2,3,6,11]:
    plt.plot(joints_f[i][0], joints_f[i][1], marker='.', color="red")

  plt.subplot(224)
  plt.imshow((((inp_s / 2.)+0.5)*255).astype(int))
  plt.plot(r_waist_s[0],r_waist_s[1], marker='.', color="blue")
  plt.plot(l_waist_s[0],l_waist_s[1], marker='.', color="blue")
  plt.plot(r_neck_scaled_s[0],r_neck_scaled_s[1],marker = ".", color = 'blue')
  plt.plot(l_neck_scaled_s[0],l_neck_scaled_s[1],marker = ".", color = 'blue')
  plt.plot(joints_s[12][0], joints_s[12][1],marker = ".", color = 'red')
  for i in [2,3]:
    plt.plot(joints_s[i][0], joints_s[i][1], marker='.', color="red")
  # plt.show()
  plt.savefig('..\\..\\out\\FinalFeaturePoints.jpg')
  
  
  ## Measurement ##
  print("Height in pixels in front view: "+ str(ht_p_f))
  print("Height in pixels in side view: "+ str(ht_p_s))
  lpp_f = ht / ht_p_f
  lpp_s = ht / ht_p_s
  print("Length per pixel in front image (in cm): "+str(lpp_f))
  print("Length per pixel in side image (in cm): "+str(lpp_s))

  cuff = (dist(right_wrist_r[0],right_wrist_r[1],right_wrist_l[0],right_wrist_l[1]) + dist(left_wrist_r[0],left_wrist_r[1],left_wrist_l[0],left_wrist_l[1]))/2 * lpp_f
  cuff_c = cuff * wrist_factor

  print("Height: "+str(ht)+" cm")
  shoulder, arm, waist_f, neck_f = front_measurement(joints_f, r_waist_f, l_waist_f, r_neck_scaled_f, l_neck_scaled_f, lpp_f, arm_scale)
  waist_s, neck_s = side_measurement(joints_s, r_waist_s, l_waist_s, r_neck_scaled_s, l_neck_scaled_s, lpp_s)
  waist_c = circumference("Waist", waist_f/2, waist_s/2)
  neck_c = circumference("Neck", neck_f/2, neck_f/2)
  print("Wrist Front width: "+str(cuff)+ " cm")
  print("Cuff cirumference: "+str(cuff_c)+ " cm")

  ## Feedback ##
  fb = ""
  parts = ""
  if (waist_c > ht * 0.8) or (waist_c < ht * 0.3):
    parts += "Waist\n"
  if (neck_c > ht * 0.45) or (neck_c < ht * 0.10):
    parts += "Neck\n"
  if (cuff_c > ht * 0.2) or (cuff_c < ht * 0.05):
    parts += "Wrist\n"
  if parts != "":
    fb += "Please ensure that the following are clearly visible and distinguishable: " + parts
  if fb != "":
    fb += "Ignore if the feature points seem correct." 

  ## JSON output ##
  out = {}
  out['Height'] = ht
  out['Waist'] = waist_c
  out['Neck'] = neck_c
  out['Cuff'] = cuff_c
  out['Shoulder'] = shoulder
  out['Arm'] = arm
  out['feedback'] = fb
  with open('..\\..\\out\\output.json', 'w') as f:
      json.dump(out, f)

  end_bot = time.time()
  print("Output Exported... \nTotal Time taken: " + str(end_bot-start_top)+ " seconds.")


if __name__ == "__main__":
  ## Loading Models ##
  start_load = time.time()
  seg_model = segmentation_model()
  hmr_model = HMRmodel()
  ## Renderer ## - optional
  # from visualise.trimesh_renderer import TrimeshRenderer
  # renderer = TrimeshRenderer()
  end_load = time.time()
  print("Loaded Models... Time taken: " + str(end_load-start_load)+ " seconds.")
  while True:
    ht = input("Enter height in cm: ")
    if ht == "0":
      break
    process(float(ht))
