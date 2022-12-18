import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from SingleAgentB import TestEnv_Both, numberToBase
import omegaconf
import argparse

#source: https://stackoverflow.com/questions/13055884/parsing-math-expression-in-python-and-solving-to-find-an-answer
def parse_number(x):
    operators = set('+-*/')
    op_out = []    #This holds the operators that are found in the string (left to right)
    num_out = []   #this holds the non-operators that are found in the string (left to right)
    buff = []
    for c in x:  #examine 1 character at a time
        if c in operators:  
            #found an operator.  Everything we've accumulated in `buff` is 
            #a single "number". Join it together and put it in `num_out`.
            num_out.append(''.join(buff))
            buff = []
            op_out.append(c)
        else:
            #not an operator.  Just accumulate this character in buff.
            buff.append(c)
    num_out.append(''.join(buff))
    return num_out,op_out

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_operation", help="The calculation to do, e.g 5+10+25-5-5", default="5+10+25-5-5")
args = parser.parse_args()

input_string = args.input_operation
numbers, operators = parse_number(input_string)

cfg = omegaconf.OmegaConf.load("config_params.yaml")
env = TestEnv_Both(cfg.env_params)
path = "best_model.zip"
model = MaskablePPO.load(path, env=env)


env.list_number= [int(el) for el in numbers]
env.list_operation= [1 if el == "+" else 0 for el in operators]

gif_list = []

action_dict = {0: 'move_and_slide',
                1: 'left',
                2: 'right',
                3: 'down',
                4: 'up',
                5: 's_right',
                6: 's_left',
                7: 'submit'}

obs = env.reset()
for i in range(1000): #while(True):
    action, _state = model.predict(obs, deterministic=True, action_masks = get_action_masks(env))
    obs, reward, done, info = env.step(action)
    
    img_width = 50
    img_height = 50
    obs_img_shape = (img_width * env.ext_shape[1], img_height * env.ext_shape[0])
    ext_img_shape = (img_width * env.ext_shape[1], img_height * env.ext_shape[0])
    im = Image.fromarray(obs['space1'][-1][-1]*255)

    stuff = (np.rot90((1-env.ext_repr.externalrepresentation), 1))
    im = Image.fromarray(stuff *255).resize( tuple(el*50 for el in env.ext_repr.externalrepresentation.shape), resample=0)
    im = im.convert('RGB')

    draw = ImageDraw.Draw(im)
    leftUpPoint = (env.signpost*50+11, 11)
    rightDownPoint = (env.signpost*50+41, 41)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=(255,192,203))


    x = env.fingerlayer.pos_x
    y = env.fingerlayer.pos_y
    leftUpPoint = (y*50+11, (5-x)*50+11)
    rightDownPoint = (y*50+41, (5-x)*50+41)
    twoPointList = [leftUpPoint, rightDownPoint]
    draw.ellipse(twoPointList, fill=(255,192,203))

    im = add_margin(im, 0, 0, 0, 50, (169,169,169))
    im = add_margin(im, 5, 5, 5, 5, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    top_left = (y*50+4, 0)
    bottom_right = (y*50+107, 310)

    outline_width = 7
    draw.rectangle((top_left, bottom_right), width=outline_width, outline=(255,0, 0))
    action_img = Image.new(mode="RGB", size=(100, 300))
    draw = ImageDraw.Draw(action_img)
    top_left = (0, action*25)
    bottom_right = (100, action*25+25)
    draw.rectangle((top_left, bottom_right),fill =(220,220,220))
    font = PIL.ImageFont.truetype("arial.ttf", 13)
    for key in action_dict:
        draw.text((0, key*25),action_dict[key],(192,192,192),font=font)
    draw.text((0, action*25),action_dict[action],(0,0,255),font=font)
    draw.text((0, 228),"Observed digit",(255,255,255),font=font)
    draw.text((0, 243),str(obs['space0a']),(255,255,255),font=font)
    if env.is_addition:
        draw.text((0, 263),"addition",(255,255,255),font=font)
    else:
        draw.text((0, 263),"subtraction",(255,255,255),font=font)
    action_img
    action_img = add_margin(action_img, 5, 5, 5, 5, (255, 255, 255))

    new_image = Image.new('RGB',(im.size[0]+action_img.size[0], im.size[1]), (250,250,250))
    new_image.paste(im,(0,0))
    new_image.paste(action_img,(im.size[0],0))

    gif_list.append(new_image)
    
    if action == 7:
        #print("addendi base 10:", env.addendum_1,env.addendum_2)
        print("addendi:", numberToBase(env.addendum_1,5), numberToBase(env.addendum_2,5))
        #print("risultato base 10:", env.n_objects)
        print("totale:", numberToBase(env.n_objects, 5))
        a =  np.where((env.ext_repr.externalrepresentation == 0))[1].tolist()
        a.reverse()
        print("repr:", a)
    if done:
        break

gif_list[0].save('prova.gif', append_images=gif_list[1:], save_all=True, duration = len(gif_list)*10)

    
