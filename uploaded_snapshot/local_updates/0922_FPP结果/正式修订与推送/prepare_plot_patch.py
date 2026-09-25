from pathlib import Path
import json
base=Path(__file__).resolve().parent
s=(base.parent/'图表审阅稿/make_review.py').read_text(encoding='utf-8')
s=s.replace('The zero-count strip has no probability ordinate and supplies no upper bound.','Zero counts are labels below rank ticks, not probability ordinates.')
s=s.replace('6.1023622047','6.6929133858').replace('bottom=.14,top=.895','bottom=.225,top=.855')
s=s.replace('upper=[];lower=[];strips=[]','upper=[];lower=[]')
a=s.index('    gs=grid[1,col].subgridspec');b=s.index('    k=real.background',a)
s=s[:a]+"    ax=fig.add_subplot(grid[1,col]);lower.append(ax)\n"+s[b:]
a=s.index("    ax.tick_params(axis='x',which='both'");b=s.index('    audit_panels.extend',a)
s=s[:a]+'''    ax.set_xlim(.3,20.7);ax.set_xticks([1,5,10,15,20])
    ax.set_xlabel('Candidate rank',labelpad=48)
    if col==0:ax.set_ylabel('Conditional FPP (%)')
    ax.text(-.18,1.06,chr(100+col),transform=ax.transAxes,fontsize=10,fontweight='bold')
    zero_labels=[]
    for rank in ranks[k==0]:
        zero_labels.append(ax.text(rank,-.16,'0/4005',transform=ax.get_xaxis_transform(),
            rotation=90,rotation_mode='anchor',ha='right',va='center',fontsize=6.6,color='#333333',clip_on=False))
    fig.canvas.draw()
    ren=fig.canvas.get_renderer()
    for first,second in zip(zero_labels,zero_labels[1:]):
        assert not first.get_window_extent(ren).overlaps(second.get_window_extent(ren))
'''+s[b:]
s=s.replace('zero_strip_count','zero_label_count').replace('zero strip is nonnumeric','zero labels are outside probability coordinates')
s=s.replace("bbox_to_anchor=(.53,.987)","bbox_to_anchor=(.53,.95)")
a=s.index('bottom_handles=');b=s.index('fig.canvas.draw()',a)
s=s[:a]+'''fig.suptitle('0/4005 indicates that no background pair reached the candidate score',
             fontsize=8.5,fontweight='bold',y=.99)
bottom_handles=[Line2D([],[],color=TEAL,marker='o',mfc='white',ls='none',ms=4,label='1–19 exceedances'),Line2D([],[],color=TEAL,marker='o',ls='none',ms=4,label='20 or more')]
legend2=fig.legend(handles=bottom_handles,loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,columnspacing=1.5,handlelength=1.2)
'''+s[b:]
s=s.replace('for a in strips','for a in lower')
s=s.replace("'Unresolved' if r.zero_exceedance_unresolved", "'—' if r.zero_exceedance_unresolved")
s=s.replace("            fig.text(.029,.915,'Frozen C ranking with GWLR-FPP-01 background comparison',fontsize=10,color='#555555')","            fig.text(.029,.915,'0/4005 indicates that no background pair reached the candidate score',fontsize=10,color='#333333')")
s=s.replace('TriLens conditional FPP = k/4005. Unresolved: k = 0, not zero risk. It is not a lensing probability or annual FAR.','TriLens conditional FPP = k/4005 for nonzero counts. It is not a lensing probability or annual FAR.')
s=s.replace("                if row>0 and col==7 and cell.get_text().get_text()=='Unresolved':cell.get_text().set_color(ORANGE)\n",'')
s=s.replace("'plot_size_mm':[183,155]","'plot_size_mm':[183,170]")
s=s.replace('Separate nonnumeric strip; no upper limit assigned','Rank-aligned 0/4005 labels outside probability axes; no upper limit assigned')
s=s.replace('Fig4_FPP_review','Fig4_FPP_final').replace('Tables_candidates_Top20_review','Tables_candidates_Top20_final')
print(json.dumps('*** Begin Patch\n*** Add File: '+(base/'make_final.py').as_posix()+'\n'+''.join('+'+l+'\n' for l in s.splitlines())+'*** End Patch'))
