import random

#clarity
def prompts_clarity(how_many_prompts):
    Prompts = [
    "Increase the clarity!",
    "Can you please make this song sound more clear?",
    "Increase the clarity of this song by emphasizing treble frequencies.",
    "Make the audio clearer and more intelligible.",
    "Sharpen the overall sound.",
    "Bring more focus and definition to the details.",
    "Make the mix sound less cloudy.",
    "Tighten the articulation in the sound."
    ]
    return random.sample(Prompts, how_many_prompts)


#sparkle
#increase brightness (suppress darkness)
def prompts_brightness(how_many_prompts):
    Prompts = [
    "Can you please make this sound brighter?",
    "Increase the brightness!",
    "Make this audio sound brighter by emphasizing the high frequencies.",
    "Add some brightness to the high end.",
    "Make the sound more vivid and lively.",
    "Give the mix more shine and sparkle.",
    "Lift the treble for a more open tone.",
    "Enhance the presence of the upper frequencies."
    ]
    return random.sample(Prompts, how_many_prompts)




#increase darkness (reduce brightness)
def prompts_darkness(how_many_prompts):
    Prompts = [
    "Make this sound darker!",
    "Can you reduce the brightness, please?",
    "Make the audio darker by suppressing the higher frequencies.",
    "Bring in more low-mid richness to make the sound darker.",
    "Make the tone fuller and less sharp.",
    "Smooth out the highs with deeper low-end support.",
    "Round out the sound with more body.",
    "Soften the harshness with a warmer tone."
    ]
    return random.sample(Prompts, how_many_prompts)



#airiness
def prompts_airiness(how_many_prompts):
    Prompts = [
    "Make this sound more fresh and airy by emphasizing the high end frequencies.",
    "Make this feel more airy, please.",
    "Increase the perceived airiness, please.",
    "Give this a light sense of spaciousness by amplifying the higher frequencies.",
    "Add more air and openness to the sound.",
    "Make the audio feel more spacious and extended.",
    "Enhance the sense of space in the highs.",
    "Lift the top end for a more open character.",
    "Give the mix a breathier, more open feel."
    ]
    return random.sample(Prompts, how_many_prompts)



#boominess
def prompts_boominess(how_many_prompts):
    Prompts = [
    "Make it boom!",
    "Make this song sound more boomy by amplifying the low end bass frequencies.",
    "Increase the boominess, please!",
    "Give me more bass!",
    "Can you make this more bassy, please?",
    "Give the audio more roar and low-end power.",
    "Make the bass more impactful and solid.",
    "Add weight and depth to the bottom end.",
    "Reinforce the low frequencies for more energy.",
    "Boost the bass presence."
    ]
    return random.sample(Prompts, how_many_prompts)




#punch
def prompts_punch(how_many_prompts):
    Prompts = [
    "Give this song a punch!",
    "Make the transients sharper, please.",
    "Increase the punchiness of the song by emphasizing the transients.",
    "Make the audio more punchy and energetic.",
    "Bring back the snap and attack of transients.",
    "Add more impact and dynamic punch to the sound.",
    "Make drums and hits sound more aggressive and tight.",
    "Increase the percussive clarity and definition."
    ]
    return random.sample(Prompts, how_many_prompts)



#warmth
def prompts_warmth(how_many_prompts):
    Prompts = [
    "Can you make this song sound warmer, please?",
    "Increase the warmth, please.",
    "Emphasize the bass and low-mid frequencies to give this a more warm feel.",
    "Make the sound warmer and more inviting.",
    "Add some low-mid warmth to the mix.",
    "Soften the tone with a bit more body.",
    "Give the audio a warm analog feel.",
    "Enhance the warmth for a fuller sound."
    ]
    return random.sample(Prompts, how_many_prompts)



#vocals
def prompts_vocals(how_many_prompts):
    Prompts = [
    "Raise the level of the vocals, please.",
    "Can you amplify the vocals, please?",
    "Emphasize the vocals by raising the level of the mid frequencies specific for vocals.",
    "Bring the vocals forward in the mix.",
    "Make the voice clearer and more present.",
    "Increase the vocal presence by enhancing the midrange.",
    "Make the vocals stand out more.",
    "Strengthen the vocal clarity and focus."
    ]
    return random.sample(Prompts, how_many_prompts)



#reduce muddiness
def prompts_muddiness(how_many_prompts):
    Prompts = [
    "Can you make this song sound less muddy, please?",
    "Decrease the muddiness!",
    "Reduce the level of muddiness in this audio by lowering the low-mid frequencies.",
    "Clean up the muddiness in the low-mids.",
    "Make the mix sound less boxy and congested.",
    "Improve definition by reducing mud.",
    "Clear up the low-mid buildup.",
    "Make the audio tighter and less murky."
    ]
    return random.sample(Prompts, how_many_prompts)



#smooth out harshness
#tighten the low end

#XbandEQ even out eq
def prompts_xband(how_many_prompts):
    Prompts = [
    "Can you please correct the equalization?",
    "Improve the balance in the audio by fixing the chaotic equalizer, please.",
    "Make this sound balanced, please.",
    "Balance the EQ, please.",
    "Balance the tonal spectrum of the audio.",
    "Correct the unnatural frequency emphasis.",
    "Make the EQ curve smoother and more natural.",
    "Even out the EQ.",
    "Adjust the tonal balance for a more pleasing sound."
    ]
    return random.sample(Prompts, how_many_prompts)



#micchars
def prompts_mics(how_many_prompts):
    Prompts = [
    "This audio was recorded with a phone, can you fix that, please?",
    "Please make this sound better than a phone recording.",
    "Balance the EQ, please.",
    "Improve the balance in this song.",
    "Make the audio sound like it was recorded with a higher-quality microphone.",
    "Reduce the coloration added by the microphone.",
    "Make the tone more neutral and balanced.",
    "Improve the naturalness of the recording.",
    "Remove the harshness or boxiness from the mic coloration."
    ]
    return random.sample(Prompts, how_many_prompts)



#decompression
def prompts_compression(how_many_prompts):
    Prompts = [
    "Increase the dynamic range.",
    "Decompress the audio, please.",
    "Remove the compression, please.",
    "Can you fix the strong compression effect in this audio by expanding the dynamic range?",
    "Restore the dynamics of the audio.",
    "Make the sound less squashed and more open.",
    "Reduce the over-compression for a more natural feel.",
    "Bring back the contrast in volume.",
    "Let the audio breathe more and improve the dynamics."
    ]
    return random.sample(Prompts, how_many_prompts)




#make it louder!
def prompts_volume(how_many_prompts):
    Prompts = [
    "The volume is low, make this louder please!",
    "Can you make this sound louder, please?",
    "Increase the amplitude.",
    "Normalize the audio volume.",
    "Make the audio louder and more powerful.",
    "Increase the overall level.",
    "Boost the volume so it stands out more.",
    "Enhance the loudness without distorting the signal."
    ]
    return random.sample(Prompts, how_many_prompts)



#reverb
def prompts_reverb(how_many_prompts,reduce_prompt=False):
    Prompts = [
    "Can you remove the excess reverb in this audio, please?",
    "Please, dereverb this audio.",
    "Remove the echo!",
    "Please, reduce the strong echo in this song.",
    "Remove the church effect, please, lol.",
    "Clean this off any echoes!",
    "This song has too much of reverb present, can you please reduce it?",
    "Make the audio sound more dry and direct.",
    "Reduce the roominess or echo.",
    "Remove excess reverb and make it sound cleaner.",
    "Bring the sound closer and more focused.",
    "Tighten the spatial feel of the audio."
    ]
    if reduce_prompt:
        Prompts=Prompts[:-2]
    return random.sample(Prompts, how_many_prompts)



#clipping
def prompts_clipping(how_many_prompts):
    Prompts = [
    "This audio is clipping, can you please remove it?",
    "Can you remove the loud hissing in this song?",
    "Remove the clipping.",
    "Reduce the clipping and reconstruct the lost audio, please.",
    "Clean up the noisiness in the audio.",
    "Make the audio smoother and less distorted.",
    "Clean up the harshness in the signal.",
    "Reduce the gritty or crushed character.",
    "Fix the digital distortion.",
    "Make the sound more natural and less clipped."
    ]
    return random.sample(Prompts, how_many_prompts)


#destereo
def prompts_stereo(how_many_prompts):
    Prompts = [
    "Make it sound spacious!",
    "Can you make this audio stereo, please?",
    "Alter the left and right channels to give this recording a spatial feel.",
    "Disentangle the left and right channels to give this song a stereo feeling.",
    "Widen the stereo image.",
    "Make the audio feel more spacious and open.",
    "Add depth and separation between left and right.",
    "Enhance the stereo field for a more immersive sound.",
    "Bring a sense of space and dimension in the mix."
    ]
    return random.sample(Prompts, how_many_prompts)
