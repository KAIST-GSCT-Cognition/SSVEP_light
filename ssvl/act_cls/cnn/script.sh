for PID in 'P01' 'P02' 'P04' 'P05' 'P06' 'P07' 'P08' 'P09' 'P10' 'P11' 'P12' 'P13' 'P14' 'P15' 'P16' 'P17' 'P18' 'P20';
  do
    python run.py --platform Sc --target_hz 7hz --pid $PID
    python run.py --platform VR --target_hz 7hz --pid $PID
    python run.py --platform Sc --target_hz 10hz --pid $PID
    python run.py --platform VR --target_hz 10hz --pid $PID
    python run.py --platform Sc --target_hz 12hz --pid $PID
    python run.py --platform VR --target_hz 12hz --pid $PID
    python run.py --platform Sc --target_hz all --pid $PID
    python run.py --platform VR --target_hz all --pid $PID
  done

python run.py --platform Sc --target_hz 7hz --epochs 50
python run.py --platform VR --target_hz 7hz --epochs 50
python run.py --platform Sc --target_hz 10hz --epochs 50
python run.py --platform VR --target_hz 10hz --epochs 50
python run.py --platform Sc --target_hz 12hz --epochs 50
python run.py --platform VR --target_hz 12hz --epochs 50
python run.py --platform Sc --target_hz all --epochs 50
python run.py --platform VR --target_hz all --epochs 50