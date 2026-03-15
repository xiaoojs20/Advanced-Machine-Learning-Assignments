import time
import numpy as np


def cal_entropy(embeddings):
    probe = time.time()
    norms = np.linalg.norm(embeddings, axis=1)
    sim_mat = embeddings.dot(embeddings.T) / np.outer(norms, norms)
    entropy = np.abs(np.log(sim_mat + 1e-6))
    # print('[timer] cal_entropy={:.3f} s'.format(time.time() - probe))
    return sim_mat, entropy


class Beam:
    seqs = []
    score = 0.
    score_n = 0.  # normalized score
    score_ns = []  # log the normalized scores by each cut
    min_clip = 0.  # the length of shortest clip after a new frame cut
    key = ''

    def __init__(self, seqs):
        self.seqs = seqs
        self.key = '-'.join([str(x) for x in sorted(self.seqs)])

    def __eq__(self, other):
        return self.key == other.key  # the same seqs when sorted

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self):
        return 'cuts={}, score={:.4f}, score_n={:.4f}, min_clip={:.2f}, seqs=[{}], score_ns=[{}]'.format(
            len(self.seqs) - 1, self.score, self.score_n, self.min_clip,
            ','.join([str(x) for x in self.seqs]),
            ','.join(['{:.4f}'.format(x) for x in self.score_ns])
        )

    def __repr__(self):
        return self.__str__()


class MatSumHelper:
    def __init__(self, mat):
        N = len(mat)
        self.N = N
        self.cache = {}

        prefix = np.zeros((N+1, N+1))
        prefix[1:, 1:] = np.cumsum(np.cumsum(mat, axis=1), axis=0)
        self.prefix = prefix

    def sum(self, i, j):  # [i, j): i inclusive, j exclusive
        cache_key = i * self.N + j
        if cache_key not in self.cache:
            self.cache[cache_key] = self.prefix[j][j] - self.prefix[j][i] - self.prefix[i][j] + self.prefix[i][i]
        return self.cache[cache_key]


def keyframe_search(
        entropy_mat,
        max_cuts=300,
        min_clip=-1,
        stop_config=None,
        beam_size=5
):
    """
    the frame extraction 3.0 method (including video-cut)
    """
    N = len(entropy_mat)
    times = [0.5 * i for i in range(N + 1)]
    stop_reason = ''

    probe = time.time()
    sum_helper = MatSumHelper(entropy_mat)
    # print('[timer] sum_helper={:.3f} s'.format(time.time() - probe))

    # parse stop thres
    stop_config = stop_config or {}
    thres_abs = stop_config.get('abs', 0.001)
    thres_rel = stop_config.get('rel', 0.1)
    max_cuts = min(max_cuts, stop_config.get('max_cuts', 300))
    min_clip = max(min_clip, stop_config.get('min_clip', -1))
    # print('stop_config: abs={}, rel={}, max_cuts={}, min_clip={}'.format(thres_abs, thres_rel, max_cuts, min_clip))

    # initial beam, just one clip, first frame as keyframe
    init_beam = Beam(seqs=[0, N])
    init_beam.score = sum_helper.sum(0, N)
    init_beam.score_n = 1.
    init_beam.score_ns = [1.]
    init_beam.min_clip = times[-1]
    max_score = init_beam.score

    beams = [init_beam]
    frame_num = 1
    max_cuts = min(max_cuts, N)
    last_score_n, diff_abs, diff_rel = 1, 1, 1
    while frame_num < max_cuts:
        # logger.info('frame_num={:d}, diff_abs={:.4f}, diff_rel={:.4f}, beam0={}'.format(frame_num, diff_abs, diff_rel, beams[0]))
        new_beams = set()
        for beam in beams:  # every possible beam of keyframes
            for cut in range(1, N):  # search all frames to find next cut candidate, for beam extend
                if cut in beam.seqs:
                    continue
                new_beam = Beam(beam.seqs + [cut])  # update score and min_clip later
                # find where cut is (its prev and next keyframe/cut)
                prv = max([x for x in beam.seqs if x < cut])
                nxt = min([x for x in beam.seqs if x > cut])
                # score change: ...|prv|...|nxt|... -> ...|prv|...|cut|...|nxt|...
                score_delta = sum_helper.sum(prv, cut) + sum_helper.sum(cut, nxt) - sum_helper.sum(prv, nxt)
                new_beam.score = beam.score + score_delta
                new_beam.score_n = new_beam.score / max_score
                new_beam.score_ns = beam.score_ns + [new_beam.score_n]
                new_beam.min_clip = min(beam.min_clip, times[cut] - times[prv], times[nxt] - times[cut])
                if new_beam not in new_beams:
                    new_beams.add(new_beam)
        new_beams = list(sorted(new_beams))[:beam_size]

        # early stop
        # criteria 1: score change
        score_n = new_beams[0].score_n
        diff_abs = abs(last_score_n - score_n)
        diff_rel = diff_abs / (last_score_n + 1e-6)
        if diff_abs < thres_abs:
            stop_reason = 'thres_abs'
            break
        if diff_rel < thres_rel:
            stop_reason = 'thres_rel'
            break
        # criteria 2: min clip limit (usually for video-cut usage)
        if min_clip > 0:
            min_clip_avg = np.mean([beam.min_clip for beam in beams])
            if min_clip_avg < min_clip:
                stop_reason = 'min_clip'
                break

        # update
        beams = new_beams
        frame_num += 1
        last_score_n = score_n
        if frame_num == max_cuts:
            stop_reason = 'max_cuts'

    # for idx, beam in enumerate(beams):
    #     logger.info('frame_num={:d}/{:d}, diff_abs={:.4f}, diff_rel={:.4f}, beam{}={}'.format(frame_num, idx, diff_abs, diff_rel, idx, beam))
    keyframe_indices = list(sorted(beams[0].seqs))[:-1]
    stop_info = {
        'stop_reason': stop_reason,
        'diff_abs': diff_abs,
        'diff_rel': diff_rel,
        'score_n': beams[0].score_n,
        'score_ns': beams[0].score_ns,
        'avg_entropy': entropy_mat.sum() / N ** 2
    }

    return keyframe_indices, beams[0], stop_info
