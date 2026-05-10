# 临时测试 madmom 的正确调用方式
import warnings
warnings.filterwarnings("ignore")

def test_madmom(music_path):
    try:
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        act = RNNDownBeatProcessor()(music_path)
        print("act type:", type(act), "shape:", act.shape)
        # beats_per_bar 改成 [4] 试试
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[4])
        beats = proc(act)
        print("beats[:5]:", beats[:5])
        downbeats = [float(b[0]) for b in beats if int(b[1]) == 1]
        print("downbeats[:5]:", downbeats[:5])
    except Exception as e:
        print("错误:", e)
        import traceback
        traceback.print_exc()

test_madmom("test/music/test_music.mp3")
