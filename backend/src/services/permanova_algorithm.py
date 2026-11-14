import math,hashlib,json
from itertools import combinations
import numpy as np

def permanova_statistics(clr,labels):
    try:Z=np.asarray(clr,dtype=float)
    except (TypeError,ValueError,OverflowError):raise ValueError("CLR须为有限二维矩阵")
    if Z.ndim!=2 or len(Z)<4 or not 2<=Z.shape[1]<=8 or not np.isfinite(Z).all():raise ValueError("至少4行、2–8维有限CLR")
    if len(labels)!=len(Z) or any(not isinstance(g,str) or not g for g in labels):raise ValueError("每行须有非空字符串组标签")
    groups=sorted(set(labels));n=len(Z);k=len(groups)
    if not 2<=k<=5:raise ValueError("要求2–5个预定义组")
    labels=np.asarray(labels);indices={g:np.flatnonzero(labels==g) for g in groups}
    if any(len(i)<2 for i in indices.values()):raise ValueError("每组至少2个独立样本")
    D=Z-Z[0];center=np.array([math.fsum(D[:,j])/n for j in range(Z.shape[1])]);U=D-center;scale=float(np.max(np.abs(U)))
    if scale==0:raise ValueError("总CLR平方和SST=0，PERMANOVA未定义")
    V=U/scale;sst=math.fsum(float(v*v) for v in V.flat);within_terms=[];summaries=[]
    for group,ix in indices.items():
        G=V[ix];shift=G-G[0];offset=np.array([math.fsum(shift[:,j])/len(ix) for j in range(Z.shape[1])]);residual=shift-offset
        within=math.fsum(float(v*v) for v in residual.flat);within_terms.append(within)
        original=Z[ix];delta=original-original[0];mean=original[0]+np.array([math.fsum(delta[:,j])/len(ix) for j in range(Z.shape[1])]);distances=np.sqrt(np.sum(residual**2,axis=1))*scale
        exp=np.exp(mean-np.max(mean));composition_center=exp/exp.sum()
        summaries.append({"group":group,"n":len(ix),"mean_clr":mean.tolist(),"aitchison_center":composition_center.tolist(),"mean_distance_to_centroid":float(distances.mean()),"max_distance_to_centroid":float(distances.max())})
    ssw=math.fsum(within_terms)
    ssb=sst-ssw
    if ssb<0 and abs(ssb)<=64*np.finfo(float).eps*sst:ssb=0.
    if ssb<0:raise ValueError("平方和分解数值失败")
    R2=min(1.,max(0.,ssb/sst));F=None;status='unbounded' if ssw==0 else 'finite'
    if ssw>0:
        F=(ssb/(k-1))/(ssw/(n-k))
        if not np.isfinite(F):F=None;status='numerical_overflow'
    return {"n":n,"k":k,"df_between":k-1,"df_within":n-k,"SST":sst*scale*scale,"SSW":ssw*scale*scale,"SSB":ssb*scale*scale,"R2":R2,"F":F,"F_status":status,"groups":summaries,"overall_mean_clr":(Z[0]+center).tolist(),"scale":scale,"method_note":"CLR欧氏平方和；锚定中心、相对缩放及math.fsum组SSW汇总；不用绝对SST阈值。"}

def upper_tail_p(observed,null):
    values=np.asarray(null,dtype=float)
    if not np.isfinite(observed) or not 0<=observed<=1 or values.ndim!=1 or len(values)==0 or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError("R²须为[0,1]有限值且null非空")
    tolerance=64*np.finfo(float).eps*max(1.,abs(float(observed)),float(np.max(np.abs(values))))
    threshold=float(observed-tolerance);extreme=int(np.sum(values>=threshold))
    return {"p":(extreme+1)/(len(values)+1),"extreme":extreme,"resolution":1/(len(values)+1),"comparison_threshold":threshold,"tie_tolerance":float(tolerance),"tail":"R²上尾含并列，+1；并列保护为64×机器epsilon×max(1,观测与null绝对R²)，不用于SST拒绝。"}

def permutation_test(clr,labels,permutations=199,seed=2026):
    if type(permutations) is not int or not 199<=permutations<=999:raise ValueError("置换数须为199–999整数")
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError("种子须为非负32位整数")
    if len(labels)*permutations>150000:raise ValueError("样本数×置换数不能超过150000")
    observed=permanova_statistics(clr,labels);rng=np.random.default_rng(seed);null=[]
    for _ in range(permutations):null.append(permanova_statistics(clr,rng.permutation(labels).tolist())['R2'])
    return {**observed,"null_R2":null,"permutations":permutations,**upper_tail_p(observed['R2'],null)}

def _pair_seed(root_seed,group_a,group_b):
    digest=hashlib.sha256(json.dumps([root_seed,group_a,group_b],ensure_ascii=False,separators=(",",":")).encode('utf-8')).digest()
    return int.from_bytes(digest[:4],'big')&0x7fffffff

def _constant_rows(rows):
    return len(rows)>0 and bool(np.all(rows==rows[0]))

def _holm(pairs):
    testable=[p for p in pairs if p['p_raw'] is not None]
    m=len(testable);running=0.
    for rank,index in enumerate(sorted(range(m),key=lambda i:(testable[i]['p_raw'],i))):
        running=max(running,min(1.,(m-rank)*testable[index]['p_raw']))
        testable[index]['p_adjusted']=running
        testable[index]['rank']=rank+1
    return m

def pairwise_permutation_tests(clr,labels,permutations=199,seed=2026):
    if type(permutations) is not int or not 199<=permutations<=999:raise ValueError("置换数须为199–999整数")
    if type(seed) is not int or not 0<=seed<=2147483647:raise ValueError("种子须为非负32位整数")
    try:Z=np.asarray(clr,dtype=float)
    except (TypeError,ValueError,OverflowError):raise ValueError("CLR须为有限二维矩阵")
    if Z.ndim!=2 or len(Z)<4 or not 2<=Z.shape[1]<=8 or not np.isfinite(Z).all():raise ValueError("至少4行、2–8维有限CLR")
    if len(labels)!=len(Z) or any(not isinstance(g,str) or not g for g in labels):raise ValueError("每行须有非空字符串组标签")
    labels=np.asarray(labels);groups=sorted(set(labels.tolist()))
    if not 2<=len(groups)<=5:raise ValueError("要求2–5个预定义组")
    indices={g:np.flatnonzero(labels==g) for g in groups}
    pairs=[]
    for group_a,group_b in combinations(groups,2):
        ia,ib=indices[group_a],indices[group_b]
        entry={"group_a":group_a,"group_b":group_b,"n_a":int(len(ia)),"n_b":int(len(ib)),"n":int(len(ia)+len(ib)),
               "permutations":permutations,"pair_seed":_pair_seed(seed,group_a,group_b),"testable":False,"reason":"",
               "R2":None,"F":None,"F_status":None,"p_raw":None,"p_adjusted":None,"rank":None}
        if len(ia)<2 or len(ib)<2:
            entry["reason"]="每组至少2个独立样本，样本不足，不可检验";pairs.append(entry);continue
        constant=[group for group,ix in ((group_a,ia),(group_b,ib)) if _constant_rows(Z[ix])]
        if constant:
            entry["reason"]="组内CLR恒定（"+"、".join(constant)+"），不可检验";pairs.append(entry);continue
        subset=np.vstack([Z[ia],Z[ib]]);subset_labels=[group_a]*len(ia)+[group_b]*len(ib)
        try:observed=permanova_statistics(subset.tolist(),subset_labels)
        except ValueError as e:
            entry["reason"]=str(e);pairs.append(entry);continue
        centered=subset-subset.mean(axis=0);sst=math.fsum(float(v*v) for v in centered.flat)
        if not sst>0:entry["reason"]="两组合并CLR平方和SST=0，不可检验";pairs.append(entry);continue
        rng=np.random.default_rng(entry["pair_seed"]);null=[];has_a=np.zeros(len(subset),dtype=bool);has_a[:len(ia)]=True
        for _ in range(permutations):
            mask=has_a[rng.permutation(len(subset))];mean_a=centered[mask].mean(axis=0);mean_b=centered[~mask].mean(axis=0)
            ssb=len(ia)*float(mean_a@mean_a)+len(ib)*float(mean_b@mean_b)
            null.append(min(1.,max(0.,ssb/sst)))
        tail=upper_tail_p(observed['R2'],null)
        entry.update({"testable":True,"R2":observed['R2'],"F":observed['F'],"F_status":observed['F_status'],
                      "p_raw":tail['p'],"extreme":tail['extreme'],"resolution":tail['resolution']})
        pairs.append(entry)
    family_size=_holm(pairs)
    return {"method":"holm","method_label":"Holm逐步降序，控制族错误率","family_size":family_size,"pairs_total":len(pairs),
            "testable_pairs":family_size,"untestable_pairs":len(pairs)-family_size,"permutations":permutations,"seed":seed,
            "pairs":pairs,
            "correction_scope":"运行级固定：同时校正全部可检验工艺对，不随查看或切换工艺对改变。",
            "note":"每对只在自身两组样本子集内独立置换标签并派生独立种子；组内CLR恒定或样本不足的对标为不可检验，p留空且不参与Holm显著性排序；同一种子重算一致。"}
