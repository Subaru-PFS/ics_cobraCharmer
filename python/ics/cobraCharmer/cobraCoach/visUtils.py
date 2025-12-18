
import numpy as np
from ics.cobraCharmer.cobraCoach import visDianosticPlot
import psycopg2
from sqlalchemy import create_engine
#from opdb import opdb
import pandas as pd
import re
from pfs.utils.fiberids import FiberIds

def parseFpsLog(pfsVisit, logPattern='/data/logs/actors/fps/202?-*-*.log'):
    """
    從 FPS log 中擷取指定 visit 的資訊
    
    Parameters:
    -----------
    pfsVisit : int
        PFS visit ID (e.g., 134301)
    logPattern : str
        Log 檔案的搜尋路徑 pattern (default: '/data/logs/actors/fps/202?-*-*.log')
    
    Returns:
    --------
    dict : 包含以下欄位的字典
        - 'timestamp': 命令時間戳記
        - 'designId': Design ID
        - 'iteration': 迭代次數
        - 'tolerance': 收斂容忍度
        - 'exptime': 曝光時間
        - 'visit': Visit ID
        - 'log_file': 找到的 log 檔案路徑
        - 'cobras_left': 各階段剩餘的 cobra 數量列表 (例如 ['2107 left', '1983 left'])
        - 'full_log_section': 完整的 log 區段內容
    
    Example:
    --------
    >>> result = parseFpsLog(134301)
    >>> print(result['cobras_left'])
    ['2107 left', '1983 left', '1876 left']
    >>> print(result['designId'])
    '18887845828571423'
    """
    import subprocess
    import re
    import glob as glob_module
    
    # Step 1: 用 grep 找出包含 visit 的 log 檔案
    grep_cmd = f"grep -l '{pfsVisit},Preparing' {logPattern}"
    
    try:
        output = subprocess.check_output(grep_cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        log_files = output.strip().split('\n')
        
        if not log_files or not log_files[0]:
            print(f"Warning: No log file found for visit {pfsVisit}")
            return None
        
        # 使用第一個找到的檔案
        log_file = log_files[0]
            
    except subprocess.CalledProcessError as e:
        print(f"Error: No matching log found for visit {pfsVisit}")
        return None
    
    # Step 2: 讀取整個 log 檔案內容
    result = {
        'visit': pfsVisit,
        'log_file': log_file,
        'cobras_left': [],
        'left_lines': [],
        'full_log_section': ''
    }
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None
    
    # Step 3: 找到包含 moveToPfsDesign 和 visit 的命令行
    cmd_pattern = f"moveToPfsDesign.*visit={pfsVisit}"
    cmd_match = re.search(cmd_pattern, log_content)
    
    if not cmd_match:
        print(f"Warning: No moveToPfsDesign command found for visit {pfsVisit}")
        return None
    
    # Step 4: 解析命令參數
    cmd_line = log_content[cmd_match.start():cmd_match.end() + 200]  # 取多一點以確保完整
    
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}Z)', cmd_line)
    if timestamp_match:
        result['timestamp'] = timestamp_match.group(1)
    
    designId_match = re.search(r'designId=(\w+)', cmd_line)
    if designId_match:
        result['designId'] = designId_match.group(1)
    
    iteration_match = re.search(r'iteration=(\d+)', cmd_line)
    if iteration_match:
        result['iteration'] = int(iteration_match.group(1))
    
    tolerance_match = re.search(r'tolerance=([\d.]+)', cmd_line)
    if tolerance_match:
        result['tolerance'] = float(tolerance_match.group(1))
    
    exptime_match = re.search(r'exptime=([\d.]+)', cmd_line)
    if exptime_match:
        result['exptime'] = float(exptime_match.group(1))
    
    # Step 5: 找到這個命令的區段（從命令開始到下一個命令或檔案結尾）
    start_pos = cmd_match.start()
    
    # 找下一個 new cmd: 作為結束點（通常是下一個命令）
    next_cmd_pattern = r'new cmd:'
    next_cmd_match = re.search(next_cmd_pattern, log_content[start_pos + 500:])
    if next_cmd_match:
        end_pos = start_pos + 500 + next_cmd_match.start()
    else:
        # 如果沒有下一個命令，取後面 50000 個字元（足夠涵蓋整個執行過程）
        end_pos = min(start_pos + 50000, len(log_content))
    
    section = log_content[start_pos:end_pos]
    result['full_log_section'] = section
    
    # Step 6: 找出所有 "XXX left" pattern
    left_pattern = r'(\d+)\s+left'
    matches = re.findall(left_pattern, section)
    result['cobras_left'] = [f"{num} left" for num in matches]
    
    # 找出包含 "left" 的完整行
    left_lines = []
    for line in section.split('\n'):
        if re.search(r'\d+\s+left', line):
            left_lines.append(line.strip())
    result['left_lines'] = left_lines
    
    return result


def parseFpgaLog(runDir, logFile='fpgaProtocol.log', return_dataframe=False, iteration=None):
    """
    從 FPGA log 中擷取 cobra 操作資訊
    
    Parameters:
    -----------
    runDir : str
        Run directory (e.g., '20251117_001')
    logFile : str
        Log 檔案名稱 (default: 'fpgaProtocol.log')
    return_dataframe : bool
        如果為 True，返回 pandas DataFrame；如果為 False，返回 dict (default: False)
    iteration : int, optional
        如果指定，只返回該特定 iteration 的資料 (1-based, 第一筆是 iteration=1)
        如果為 None，返回所有 iterations
    
    Returns:
    --------
    dict or DataFrame : 
        如果 return_dataframe=False，返回包含以下欄位的字典：
        - 'run_dir': Run directory
        - 'log_file': Log 檔案完整路徑
        - 'cmd_runs': CMD run 命令列表，每個包含：
            - 'cmdNum': 命令編號
            - 'nCobras': 操作的 cobra 數量
            - 'timeLimit': 時間限制
            - 'interleave': Interleave 參數
            - 'timestamp': 時間戳記
            - 'cobras': Cobra 操作詳細資料列表
        - 'total_operations': 總操作次數
        - 'nCobras_history': nCobras 的歷史變化
        
        如果 return_dataframe=True，返回 pandas DataFrame，包含所有 cobra 操作資料
    
    Example:
    --------
    >>> result = parseFpgaLog('20251117_001')
    >>> print(result['nCobras_history'])
    [2137, 2137, 2137, 2006, 1670, 1166, 798, 554]
    >>> print(len(result['cmd_runs']))
    8
    
    >>> # 只取第 3 個 iteration
    >>> result = parseFpgaLog('20251117_001', iteration=3)
    >>> print(len(result['cmd_runs']))
    1
    >>> print(result['cmd_runs'][0]['cobras'][0])
    {'module_id': 1, 'positioner_id': 1, 
     'theta_status': 1, 'theta_dir': 'ccw', 'theta_steps': 2612, 'theta_ontime': 52.0, 'theta_freq': 1388.0,
     'phi_status': 1, 'phi_dir': 'cw', 'phi_steps': 479, 'phi_ontime': 31.0, 'phi_freq': 3521.0}
    
    >>> df = parseFpgaLog('20251117_001', return_dataframe=True, iteration=3)
    >>> print(df['iteration'].unique())
    [3]
    """
    import re
    import os
    
    # 建構完整路徑
    log_path = f'/data/MCS/{runDir}/logs/{logFile}'
    
    if not os.path.exists(log_path):
        print(f"Error: FPGA log file not found: {log_path}")
        return None
    
    result = {
        'run_dir': runDir,
        'log_file': log_path,
        'cmd_runs': [],
        'total_operations': 0,
        'nCobras_history': []
    }
    
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading FPGA log file {log_path}: {e}")
        return None
    
    # 找出所有 CMD run 命令
    cmd_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}Z).*CMD run cmdNum=\s*(\d+)\s+nCobras=\s*(\d+)\s+timeLimit=\s*(\d+)\s+interleave=\s*(\d+)'
    cmd_matches = list(re.finditer(cmd_pattern, log_content))
    
    result['total_operations'] = len(cmd_matches)
    
    # 如果指定了 iteration，檢查是否有效
    if iteration is not None:
        if iteration < 1 or iteration > len(cmd_matches):
            print(f"Error: iteration {iteration} out of range (1-{len(cmd_matches)})")
            return None
    
    # 解析每個 CMD run
    for i, cmd_match in enumerate(cmd_matches):
        current_iteration = i + 1  # iteration 從 1 開始
        
        # 如果指定了 iteration，只處理該 iteration
        if iteration is not None and current_iteration != iteration:
            continue
        
        timestamp, cmdNum, nCobras, timeLimit, interleave = cmd_match.groups()
        
        cmd_info = {
            'cmdNum': int(cmdNum),
            'nCobras': int(nCobras),
            'timeLimit': int(timeLimit),
            'interleave': int(interleave),
            'timestamp': timestamp,
            'cobras': []
        }
        
        result['nCobras_history'].append(int(nCobras))
        
        # 找出這個 CMD run 後面的所有 cobra 操作
        start_pos = cmd_match.end()
        
        # 找下一個 CMD run 或檔案結尾
        if i + 1 < len(cmd_matches):
            end_pos = cmd_matches[i + 1].start()
        else:
            end_pos = len(log_content)
        
        section = log_content[start_pos:end_pos]
        
        # 解析 cobra 操作行
        # 格式: run cobra:  1  1  Theta: 1 ccw   2612  52.0 1388.0  Phi: 1  cw    479  31.0 3521.0
        #                  │  │         │  │      │     │    │            │  │     │    │    │
        #              module pos   status dir  steps ontime freq      status dir steps ontime freq
        cobra_pattern = r'run cobra:\s+(\d+)\s+(\d+)\s+Theta:\s+(\d+)\s+(c?cw)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+Phi:\s+(\d+)\s+(c?cw)\s+(\d+)\s+([\d.]+)\s+([\d.]+)'
        
        for cobra_match in re.finditer(cobra_pattern, section):
            module_id, positioner_id, theta_status, theta_dir, theta_steps, theta_ontime, theta_freq, \
            phi_status, phi_dir, phi_steps, phi_ontime, phi_freq = cobra_match.groups()
            
            cobra_data = {
                'module_id': int(module_id),
                'positioner_id': int(positioner_id),
                'theta_status': int(theta_status),
                'theta_dir': theta_dir,
                'theta_steps': int(theta_steps),
                'theta_ontime': float(theta_ontime),
                'theta_freq': float(theta_freq),
                'phi_status': int(phi_status),
                'phi_dir': phi_dir,
                'phi_steps': int(phi_steps),
                'phi_ontime': float(phi_ontime),
                'phi_freq': float(phi_freq)
            }
            
            cmd_info['cobras'].append(cobra_data)
        
        result['cmd_runs'].append(cmd_info)
    
    # 如果要返回 DataFrame
    if return_dataframe:
        try:
            import pandas as pd
            
            data_rows = []
            for i, cmd in enumerate(result['cmd_runs'], 1):
                # 如果有指定 iteration，使用該值；否則使用順序編號
                iter_num = iteration if iteration is not None else i
                
                for cobra in cmd['cobras']:
                    data_rows.append({
                        'iteration': iter_num,
                        'timestamp': cmd['timestamp'],
                        'cmdNum': cmd['cmdNum'],
                        'nCobras': cmd['nCobras'],
                        'timeLimit': cmd['timeLimit'],
                        'interleave': cmd['interleave'],
                        'board_id': cobra['module_id'],
                        'positioner_id': cobra['positioner_id'],
                        'theta_status': cobra['theta_status'],
                        'theta_dir': cobra['theta_dir'],
                        'theta_steps': cobra['theta_steps'],
                        'theta_ontime': cobra['theta_ontime'],
                        'theta_freq': cobra['theta_freq'],
                        'phi_status': cobra['phi_status'],
                        'phi_dir': cobra['phi_dir'],
                        'phi_steps': cobra['phi_steps'],
                        'phi_ontime': cobra['phi_ontime'],
                        'phi_freq': cobra['phi_freq']
                    })
            
            return pd.DataFrame(data_rows)
        except ImportError:
            print("Warning: pandas not available, returning dict instead")
            return result
    
    return result


def extractNotDoneCobraCounts(result):
    """從結果中提取 cobra 數量的時間序列"""
    if not result or not result.get('cobras_left'):
        return []
    
    counts = []
    for left_str in result['cobras_left']:
        # 從 "2107 left" 提取數字 2107
        match = re.search(r'(\d+)', left_str)
        if match:
            counts.append(int(match.group(1)))
    
    return counts


def loadGrandFiberMap(filepath='/software/devel/chyan/pfs_utils/data/fiberids/grandfibermap.txt', 
                      return_dataframe=True):
    
    gfm = pd.DataFrame(FiberIds().data)
    gfm = gfm[(gfm.cobraId < 3000) & (gfm.cobraId > 0)]
    fgfm = gfm[['cobraId', 'fiberId', 'boardId', 'cobraInBoardId']].sort_values(['boardId', 'cobraInBoardId'])

    return fgfm.reset_index(drop=True)

def visGetSoppedCobraFromLogs(visit):
    """
        Get number of finished cobras from log files
    """
    runDir = visDianosticPlot.findRunDir(visit) 

    movfile = f'/data/MCS/{runDir}/data/moves.npy'

    mov = np.load(movfile)
    try:
        maxIteration = mov.shape[2]
    except:
        maxIteration = mov.shape[-1]  

    logCobraStopped = np.concatenate((
        [mov[0,:,0].shape[0]],
        extractNotDoneCobraCounts(parseFpsLog(visit))[-maxIteration+1:]
        ))
    log_finished = logCobraStopped[0] - logCobraStopped
    return log_finished

def getMaxIterationFromRunDir(runDir):
    try:
        path=f'/data/MCS/{runDir}/data/'
        movfile = path+'moves.npy'
        mov=np.load(movfile)
        iteration = mov.shape[-1]
    except:
        iteration = None

    return iteration


def getMeanTargetDist(runDir, iteration):
    
    
    pfsVisitID = visDianosticPlot.findVisit(runDir)
    conn = psycopg2.connect("dbname='opdb' host='db-ics' port=5432 user='pfs'") 
    engine = create_engine('postgresql+psycopg2://', creator=lambda: conn)

    fiberData = pd.read_sql(f'''
        SELECT DISTINCT 
            fiber_id, pfi_center_final_x_mm, pfi_center_final_y_mm, 
            pfi_nominal_x_mm, pfi_nominal_y_mm
        FROM 
            pfs_config_fiber
        WHERE
            pfs_config_fiber.visit0 = %(visit0)s
        -- limit 10
    ''', engine, params={'visit0': pfsVisitID})

    if len(fiberData) != 0:
        from pfs.utils.fiberids import FiberIds
        fid=FiberIds()
        fiberData['cobra_id']=fid.fiberIdToCobraId(fiberData['fiber_id'].values)
        fiberData=fiberData.sort_values('cobra_id')
        df = fiberData.loc[fiberData['cobra_id'] != 65535]
        unassigned_rows = df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].isna().all(axis=1)]
        unassigned_cobraIdx =  unassigned_rows['cobra_id'].values - 1 

        assigned_row= df[df[['pfi_nominal_x_mm', 'pfi_nominal_y_mm']].notna().all(axis=1)]
        assigned_cobraIdx =  assigned_row['cobra_id'].values - 1 

        targetFromDB = df['pfi_nominal_x_mm'].values+df['pfi_nominal_y_mm'].values*1j

        targets = targetFromDB
    else:
        # conraId of the disabled Cobras
        disabled = np.array([46,   49,  172,  192,  343,  346,  360,  442,  492,  647,  737,
                753,  798,  820,  852,  948, 1149, 1207, 1209, 1302, 1459, 1493,
               1519, 1538, 1579, 1636, 1652, 1723, 1789, 1790, 1791, 1824, 1835,
               1881, 1902, 2052, 2351, 2379])
        assigned_cobraIdx= np.array([i for i in range(2394) if i not in disabled])        
    
        tarfile=f'/data/MCS/{runDir}/data/targets.npy'
        targets=np.load(tarfile)
        
    moveFile = f'/data/MCS/{runDir}/data/moves.npy'
    mov = np.load(moveFile)
    try:
        maxMove = mov.shape[2]
    except:
        maxMove = mov.shape[-1]
        
    frameid = pfsVisitID*100+12

    db=opdb.OpDB(hostname='db-ics', port=5432,
           dbname='opdb',username='pfs')

    match = db.bulkSelect('cobra_match','select * from cobra_match where '
              f'pfs_visit_id = {pfsVisitID} and iteration = {iteration}').sort_values(by=['cobra_id']).reset_index()
    
    if len(fiberData) != 0:
        dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].real)**2+
                                (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets[assigned_cobraIdx].imag)**2)
    else:
        dist=np.sqrt((match['pfi_center_x_mm'].values[assigned_cobraIdx]-targets.real)**2+
                                (match['pfi_center_y_mm'].values[assigned_cobraIdx]-targets.imag)**2) 
    conn.close()
    db.close()
    return np.mean(dist), np.median(dist), np.percentile(dist, 75), np.percentile(dist, 95)