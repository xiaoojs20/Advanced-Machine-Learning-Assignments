import time
from video_llama.models import extract_frame

def cut(frames_hidden_state):
    frame_hidden_state = frames_hidden_state.mean(dim=1)
    vectors = frame_hidden_state.detach().cpu().numpy()
    # print('vectors: {}'.format(vectors.shape))

    sim_matrix, entropy_mat = extract_frame.cal_entropy(vectors)
    # print('entropy_mat: {}'.format(entropy_mat.shape))

    probe = time.time()
    stop_config = {'abs': 0.005, 'rel': 0.1, 'max_cuts': 8, 'min_clip': -1}
    choices, beam, stop_info = extract_frame.keyframe_search(entropy_mat=entropy_mat, beam_size=5,
                                                                stop_config=stop_config)
    # print('best_beam={}'.format(beam))
    # print('stop_info={}'.format(stop_info))
    cut_seqs = beam.seqs
    # print('choices={}'.format(choices))
    # print('[timer] extraction={:.3f} s'.format(time.time() - probe))
    
    num = len(choices)
    event_hidden_states = []
    for i in range(num - 1):
        event_hidden_states.append(frames_hidden_state[choices[i]: choices[i + 1], :, :])
    event_hidden_states.append(frames_hidden_state[choices[-1]:, :, :])

    return event_hidden_states