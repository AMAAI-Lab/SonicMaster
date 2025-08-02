

# load
# roll how many augs
# decide which augs
# do them in order / roll on order?
# create a json entry?
# add sentences...
# optional - save all forms of degradation and sentences separately OR just save the final degraded with all associated sentences... and mix it all through distribution

# normalize after finishing, with clipping - skip?
# cut signals to N (original length)



# import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
import librosa
import scipy.signal as signal
import soundfile as sf

import json, os
from glob import glob
import random

from deg_functions import *
from prompt_functions import *
from time import time
import argparse

timestart=time()

random.seed(28)
np.random.seed(28)

in_folder='/data/shared//targets'
out_folder='/mastering/samples_degrad_folder_test6'

# in_json='jan/mastering/targets_test.jsonl'
in_json='//mastering/degradchunks/tarchunk_5.jsonl'
out_json='/mastering/degraded_test9.jsonl'

mic_ir_folder='//mastering/smallpoli/irs'
rir_folder='//mastering/rirs'
fs=44100
stereo_thr=0.08


rirs = glob(rir_folder+'/*')
N_rirs = len(rirs)


degradation_groups = {
    "EQ": {
        "prob": 0.4,
        "options": {
            "xband": 7, "mic": 5, "bright": 3, "dark": 3, "airy": 2,
            "boom": 2, "clarity": 3, "mud": 3, "warm": 3, "vocal": 4
        }
    },
    "Dynamics": {
        "prob": 0.125,
        "options": {
            "comp": 2.5, "punch": 1
        }
    },
    "Reverb": {
        "prob": 0.225,
        "options": {
            "small": 0.15, "big": 0.15, "mix": 0.3, "real": 0.4
        }
    },
    "Amplitude": {
        "prob": 0.125,
        "options": {
            "clip": 3, "volume": 1
        }
    },
    "Stereo": {
        "prob": 0.125,
        "options": {
            "stereo": 1
        }
    }
}


def choose_degradation(stereo_ok=True,vocal_enable=True):
    # Pick a group based on top-level probability
    groups = list(degradation_groups.keys())
    if stereo_ok==False:
        groups.remove('Stereo')
    group_probs = [degradation_groups[g]["prob"] for g in groups]
    group = random.choices(groups, weights=group_probs, k=1)[0]
    

    # Pick an effect from the group based on internal score
    options = degradation_groups[group]["options"]

    option_names = list(options.keys())
    option_weights = list(options.values())
    if group=='EQ' and vocal_enable==False: #remove vocal degradation option for instrumental songs
        # options.remove('vocal')
        # del options['vocal']
        option_names=option_names[:-1]
        option_weights=option_weights[:-1]

    degrad = random.choices(option_names, weights=option_weights, k=1)[0]

    return group,degrad


with open(in_json, "r", encoding="utf-8") as infile, \
        open(out_json, "w", encoding="utf-8") as outfile:
    song_counter=0
    for line in infile:
        original = json.loads(line)
        original_id = original["id"]

        vocalinst = original["vocalinstrumental"] #vocal or instrumental track?
        if vocalinst=="vocal":
            vocal_enable=True
        else:
            vocal_enable=False

        reduce_prompt_reverb=False

        inpath = os.path.join(in_folder,original_id+".flac")

        degradation_counts = [1]*4 + [2]*2 + [3]*1  # 4x1, 2x2, 1x3
        single_degrad_counter = [] #to prevent multiple stereo degrad versions of the same sample...


        #load audio
        #get diff std to allow destereo
        orig_audio, sr = librosa.load(inpath,sr=fs,mono=False)
        if orig_audio.ndim==1: #expand mono to stereo
            orig_audio = np.stack((orig_audio,orig_audio))


        diff_std = np.std(orig_audio[0,:]-orig_audio[1,:])
        if diff_std>stereo_thr:
            stereo_ok=True
        else:
            stereo_ok=False

        for ver_index, count in enumerate(degradation_counts):

            audio=orig_audio
            audio = audio-np.mean(audio,axis=1,keepdims=True) #DC offset
            # degradations = set()
            final_prompt=[]
            final_alt_prompt=[]

            degrad_tracking={"EQ": [], "Dynamics": [], "Reverb": [], "Amplitude": [], "Stereo": []}
            try:

                degrad_groups = set()
                degrad_specific = set()
                while len(degrad_groups) < count:
                    # degradations.add(choose_degradation())
                    deg_group, deg_spec = choose_degradation(stereo_ok,vocal_enable)
                    if deg_group not in degrad_groups:
                        degrad_groups.add(deg_group)
                        degrad_specific.add(deg_spec)

                degrad_groups = list(degrad_groups)
                degrad_specific = list(degrad_specific)
                #degrads...
                
                while len(degrad_groups)==1 and ((degrad_groups[0]=="Stereo" and "stereo" in single_degrad_counter) or (degrad_specific[0]=="punch" and "punch" in single_degrad_counter) or (degrad_specific[0]=="clarity" and "clarity" in single_degrad_counter) or (degrad_specific[0]=="clip" and "clip" in single_degrad_counter)):
                    deg_group, deg_spec = choose_degradation(stereo_ok,vocal_enable)
                    degrad_groups[0] = deg_group
                    degrad_specific[0] = deg_spec

                #HIDDEN CLIPPING FOR REVERB AND COMPRESSOR
                hidden_clipping=False
                hc=[hidden_clipping,0]
                if "Amplitude" not in degrad_groups and ("comp" in degrad_specific or "Reverb" in degrad_groups):
                    if random.random()<0.15:
                        hidden_clipping=True


                if "Dynamics" in degrad_groups:
                    if "comp" in degrad_specific:

                        # threshold_db = random.randint(-40,-35)
                        threshold_db = random.randint(-45,-38)
                        ratio = random.randint(60,450)/10.0
                        manual_gain_db = random.randint(160,250)/10.0
                        attack_ms = random.randint(3,150)
                        release_ms = random.randint(80,250)

                        out_audio = compress_audio_file(
                        audio.T,
                        threshold_db=threshold_db,
                        ratio=ratio,
                        attack_ms=attack_ms,
                        release_ms=release_ms,
                        manual_gain_db=manual_gain_db,
                        fs=fs
                        )

                        audio=out_audio.T

                        prompt,alt_prompt = prompts_compression(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        #check for amplitude - roll for clipping...
                        degrad_tracking["Dynamics"]=["comp",[threshold_db, ratio, attack_ms, release_ms, manual_gain_db]]

                    
                    elif "punch" in degrad_specific:
                        attack_ms=3
                        release_ms=150
                        lookahead_ms=10

                        # left,lthre,lredu=reduce_punch_auto(audio.T[:,0], fs, attack_ms=3, release_ms=150, lookahead_ms=5) #different settings for left and right? make it unified
                        # right,rthre,rredu=reduce_punch_auto(audio.T[:,1], fs, attack_ms=3, release_ms=150, lookahead_ms=5)
                        # audio = np.stack((left,right))

                        audio,threshold_db,reduction_db=reduce_punch_auto_stereo(audio, fs, attack_ms=attack_ms, release_ms=release_ms, lookahead_ms=lookahead_ms)

                        # print("thre left",lthre,"/n redu left",lredu)
                        # print("thre right",rthre,"/n redu right",rredu)
                        # print("thre mid",mthre,"/n redu mid",mredu)
                        # print("thre mid",threshold_db,"/n redu mid",reduction_db)

                        # threshold_db=2
                        # reduction_db=1

                        prompt,alt_prompt = prompts_punch(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["Dynamics"]=["punch",[threshold_db, reduction_db, attack_ms, release_ms, lookahead_ms]]

                if "EQ" in degrad_groups:
                    if "xband" in degrad_specific:

                        n_bands = random.randint(8, 12)
                        center_freqs = np.geomspace(40, 16000, n_bands)
                        Q = center_freqs[-1] / (center_freqs[-1]-center_freqs[-2]) / 2
                        gains_db = np.random.randint(-6,7,n_bands)
                        audio = apply_peak_eq(audio, center_freqs, Q, gains_db, fs)

                        prompt,alt_prompt = prompts_xband(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["xband",[n_bands, gains_db.tolist()]]


                    elif "mic" in degrad_specific:
                        mic_number=np.random.randint(20) #choose the mic index
                        audio,phonename = microphone_function(audio,mic_number,mic_ir_folder,fs)

                        prompt,alt_prompt = prompts_mics(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["mic",[mic_number,phonename]]

                    elif "bright" in degrad_specific:
                        gain=random.randint(6,15)
                        audio=reduce_brightness(audio, fs, gain_db=gain)

                        prompt,alt_prompt = prompts_brightness(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["bright",[gain]]

                    elif "dark" in degrad_specific:
                        gain=random.randint(6,15)
                        audio=reduce_darkness(audio, fs, gain_db=gain)

                        prompt,alt_prompt = prompts_darkness(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["dark",[gain]]


                    elif "airy" in degrad_specific:
                        gain=random.randint(10,20)
                        audio=reduce_air(audio, fs, gain_db=gain)

                        prompt,alt_prompt = prompts_airiness(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["airy",[gain]]


                    elif "boom" in degrad_specific: 
                        gain=random.randint(10,20)
                        audio=reduce_boom(audio, fs, gain_db=gain)

                        prompt,alt_prompt = prompts_boominess(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["boom",[gain]]


                    elif "clarity" in degrad_specific:
                        order=random.randint(3,5)
                        audio=remove_clarity(audio, order, fs)

                        prompt,alt_prompt = prompts_clarity(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["clarity",[order]]


                    elif "mud" in degrad_specific:
                        gain=random.randint(6,15)
                        audio=increase_muddiness(audio, fs, gain)

                        prompt,alt_prompt = prompts_muddiness(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["mud",[gain]]

                    elif "warm" in degrad_specific:
                        gain=random.randint(6,20)
                        audio=reduce_warmth(audio, fs, gain_db=gain)

                        prompt,alt_prompt = prompts_warmth(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["warm",[gain]]

                    elif "vocal" in degrad_specific:
                        gain=random.randint(6,20)
                        out_audio=lower_vocals3(audio, fs, gain)

                        prompt,alt_prompt = prompts_vocals(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["EQ"]=["vocal",[gain]]



                if "Reverb" in degrad_groups:
                    if "real" not in degrad_specific: #then run pyroomacoustics
                        if "small" in degrad_specific:
                            sim_mode='simple'
                            a=np.array((3,3,2.5))
                            b=np.array((4,6,1.5))
                            absorption = random.random()*0.25+0.05
                            revword="small"
                        elif "big" in degrad_specific:
                            sim_mode='simple'
                            a=np.array((7,8,4))
                            b=np.array((8,10,10))
                            absorption = random.random()*0.25+0.05
                            revword="big"
                        elif "mix" in degrad_specific:
                            sim_mode='mix'
                            a=np.array((4,4,2.5)) #room size
                            b=np.array((4,3,1))
                            revword="mix"

                            how_many_abs_walls=np.random.randint(2)+1
                            which_walls=random.sample(range(6),how_many_abs_walls)

                            absmats=[]
                            for i in range(how_many_abs_walls): #absorptive walls
                                c=np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25, 0.25, 0.25, 0.30, 0.35])
                                d=np.array([0.50, 0.50, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.90, 0.95])
                                coeffs=(d-c)*np.random.random_sample(len(c))+c
                                mat = pra.Material(
                                energy_absorption={
                                    'coeffs': coeffs,
                                    'center_freqs': np.array([125,   250,   500,  1000,  2000,  4000,  8000, 12000, 16000, 20000])
                                }
                                )
                                absmats.append(mat)

                            refmats=[]
                            for i in range(6-how_many_abs_walls): #reflective walls
                                c=np.array([0.05, 0.05, 0.06, 0.08, 0.10, 0.12, 0.13, 0.15, 0.15, 0.17])
                                d=np.array([0.15, 0.15, 0.15, 0.20, 0.20, 0.22, 0.23, 0.25, 0.25, 0.27])
                                coeffs=(d-c)*np.random.random_sample(len(c))+c
                                mat = pra.Material(
                                energy_absorption={
                                    'coeffs': coeffs,
                                    'center_freqs': np.array([125,   250,   500,  1000,  2000,  4000,  8000, 12000, 16000, 20000])
                                }
                                )
                                refmats.append(mat)

                            allmats=[]
                            k=0
                            m=0
                            for i in range(6):
                                if i in which_walls:
                                    allmats.append(absmats[k])
                                    k+=1
                                    continue
                                else:
                                    allmats.append(refmats[m])
                                    m+=1
                                    continue
                                allmats.append()
                            
                            absorption = {
                                'east': allmats[0],
                                'west': allmats[1],
                                'north': allmats[2],
                                'south': allmats[3],
                                'ceiling': allmats[4],
                                'floor': allmats[5]
                            }


                        #get room size, source and mic positions
                        room_size=b*np.random.random_sample(3)+a
                        source_position=np.random.random_sample(3)*(0.8*room_size)+0.1*room_size #not too near the walls
                        mic_position=np.random.random_sample(3)*(0.8*room_size)+0.1*room_size #not too near the walls
                        source_position[2]=random.random()*room_size[2]*0.7+0.3 # source not near the ceiling, not on the floor either, at least 30 cm far
                        mic_position[2]=random.random()*room_size[2]*0.5+0.3 # mic not that high, but not near the floor either

                        audio=room_function(audio,room_size,source_position,mic_position,absorption,fs,sim_mode)

                        if revword=="mix":
                            centerfreqs=[125,   250,   500,  1000,  2000,  4000,  8000, 12000, 16000, 20000]
                            wall_materials_info = []
                            for wall_name, material in absorption.items():
                                wall_info = {
                                    'wall': wall_name,
                                    'center_freqs': centerfreqs,
                                    'absorption': material.absorption_coeffs.tolist()
                                }
                                wall_materials_info.append(wall_info)


                            degrad_tracking["Reverb"]=[revword,[list(room_size),list(source_position),list(mic_position),wall_materials_info]]
                        else:
                            degrad_tracking["Reverb"]=[revword,[list(room_size),list(source_position),list(mic_position),absorption]]

                        

                    elif "real" in degrad_specific:

                        rir_index=np.random.randint(N_rirs)
                        rir_name=os.path.basename(rirs[rir_index]).split('.')[0]
                        ir, fs = librosa.load(rirs[rir_index],sr=44100,mono=False)
                        N=len(audio[0,:])

                        start_index=min(np.argmax(ir,1))
                        ir=ir[:,start_index:]

                        if ir.shape[0]==2:

                            ch1=signal.fftconvolve(audio[0,:],ir[0,:],mode='full')
                            ch2=signal.fftconvolve(audio[1,:],ir[1,:],mode='full')

                            out_audio=np.array((ch1,ch2))
                            audio=out_audio[:,:N]
                            # print(str(rir_name),str(output_file),str(np.max(np.abs(out_audio))))

                        elif ir.shape[0]==4:
                            ch1=audio[0,:]
                            ch2=audio[1,:]
                            convolved1 = [signal.fftconvolve(ch1, ir[i,:], mode='full') for i in range(4)]
                            convolved2 = [signal.fftconvolve(ch2, ir[i,:], mode='full') for i in range(4)]
                            ch1rev = convolved1[0] + 0.5 * convolved1[1] + 0.2 * convolved1[3]
                            ch2rev = convolved2[0] + 0.5 * convolved2[1] + 0.2 * convolved2[3]
                            # ch1rev = 0.6 * convolved1[0] + 0.3 * convolved1[1] + 0.1 * convolved1[3]
                            # ch2rev = 0.6 * convolved2[0] + 0.3 * convolved2[1] + 0.1 * convolved2[3]
                            out_audio=np.array((ch1rev,ch2rev))
                            audio=out_audio[:,:N]

                            # print(str(rir_name),str(output_file),str(np.max(np.abs(out_audio))))

                        degrad_tracking["Reverb"]=["real",[rir_index,rir_name]]

                    if "Stereo" in degrad_groups or "airy" in degrad_specific:
                        reduce_prompt_reverb=True
                    prompt,alt_prompt = prompts_reverb(2,reduce_prompt_reverb)
                    final_prompt.append(prompt)
                    final_alt_prompt.append(alt_prompt)
                    reduce_prompt_reverb=False


                    


                if "Stereo" in degrad_groups:
                    audio=destereo_audio(audio)

                    prompt,alt_prompt = prompts_stereo(2)
                    final_prompt.append(prompt)
                    final_alt_prompt.append(alt_prompt)
                    degrad_tracking["Stereo"]=["stereo",["combined channels"]]

                if "Amplitude" in degrad_groups:
                    if "clip" in degrad_specific:
                        clip_opts=[2,3,5]
                        # clip_int=random.randint(3,10)
                        clip_int=random.choice(clip_opts)
                        audio=clip_audio_choice(audio,clip_int)

                        prompt,alt_prompt = prompts_clipping(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["Amplitude"]=["clip",[clip_int]]

                    if "volume" in degrad_specific:
                        # vol_mult=random.randint(1,5)/100 #0.01 to 0.05
                        # vol_mult=random.randint(1,10)/1000 #0.001 to 0.01
                        vol_opts=[0.001, 0.003, 0.01, 0.05]
                        vol_mult=random.choice(vol_opts)

                        audio=lower_volume(audio,vol_mult) 

                        prompt,alt_prompt = prompts_volume(2)
                        final_prompt.append(prompt)
                        final_alt_prompt.append(alt_prompt)
                        degrad_tracking["Amplitude"]=["volume",[vol_mult]]

                audio = audio-np.mean(audio,axis=1,keepdims=True) #DC offset

                amp=np.max(np.abs(audio))
                if hidden_clipping:
                    if amp>5:
                        hid_clip_int=5
                    elif amp>3:
                        hid_clip_int=3
                    else:
                        hid_clip_int=2
                    audio=clip_audio_choice(audio,hid_clip_int)
                    hc=[hidden_clipping,hid_clip_int]



                #NORMALIZE unless amplitude degrad...
                elif "Amplitude" not in degrad_groups and not hidden_clipping:
                    if amp>1:
                        norm_factor = random.randint(80,100)/100
                        audio = normalize(audio)*norm_factor
                    # put between 0.8 and 1.0, model will normalize all inputs when loading


                

                audio_out_name=os.path.join(out_folder,f"{original_id}_deg{ver_index+1}"+".flac")
                sf.write(audio_out_name, audio.T, samplerate=fs, format='FLAC')

                if len(degrad_groups)==1:
                    single_degrad_counter.append(degrad_specific[0])
                    # print(single_degrad_counter)

                random.shuffle(final_prompt) # shuffle the sentence order
                random.shuffle(final_alt_prompt) # shuffle the sentence order
                prompt_tgt=""
                alt_prompt_tgt=""
                for sentence in final_prompt: # connect the sentences
                    prompt_tgt += sentence + " "
                for sentence in final_alt_prompt: # connect the sentences
                    alt_prompt_tgt += sentence + " "

                prompt_tgt=prompt_tgt[:-1] #to remove the final space character
                alt_prompt_tgt=alt_prompt_tgt[:-1] #to remove the final space character

                degraded_entry = {
                **original,
                "source_id": original_id,
                "id": f"{original_id}_deg{ver_index+1}",
                "degradations": degrad_groups,
                "degradations_specifics": degrad_specific,
                "prompt": prompt_tgt,
                "alt_prompt": alt_prompt_tgt,
                "degradation_tracking": degrad_tracking,
                "hidden_clipping": hc
                }

                json.dump(degraded_entry, outfile)
                outfile.write("\n")


            except Exception as e:
                print('ERROR'+str(e))

        print(song_counter)
        song_counter+=1




print(time()-timestart)