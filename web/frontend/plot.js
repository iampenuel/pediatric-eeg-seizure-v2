/* Small, bundled canvas plotting module. Display transforms never affect inference. */
window.EEGPlot = (() => {
  const colors = {
    text: '#171717', secondary: '#505050', border: '#D8D8D3',
    trace: '#4B6170', accent: '#4F6675', selected: '#E8EDF0',
    annotation: '#E9EEF2', above: '#E9DFC8'
  };
  const mono = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Consolas, monospace';
  const sans = '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif';
  function context(canvas) {
    const dpr = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    canvas.width = Math.round(box.width * dpr); canvas.height = Math.round(box.height * dpr);
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr); ctx.font = `12px ${mono}`;
    return {ctx, width:box.width, height:box.height};
  }
  function regions(ctx, intervals, x, top, height, color) {
    ctx.fillStyle = color;
    for (const [a,b] of intervals) ctx.fillRect(x(a), top, x(b)-x(a), height);
  }
  function regionRibbons(ctx, annotations, aboveThreshold, x, left, width, top) {
    // Separate edge bands retain both interval types when regions overlap.
    // They sit outside the data area, so they never obscure samples or scores.
    ctx.save(); ctx.beginPath(); ctx.rect(left,top-8,width,7); ctx.clip();
    regions(ctx,annotations,x,top-8,3,colors.annotation);
    regions(ctx,aboveThreshold,x,top-4,3,colors.above);
    ctx.restore();
  }
  function selection(ctx, window, x, top, height) {
    if (!window) return;
    ctx.save();
    ctx.globalAlpha = .25;
    regions(ctx, [[window.start_seconds, window.end_seconds]], x, top, height, colors.selected);
    ctx.restore();
  }
  function selectionBorder(ctx, window, x, top, height) {
    if (!window) return;
    ctx.strokeStyle = colors.accent; ctx.lineWidth = 1.5;
    ctx.strokeRect(x(window.start_seconds)+.75, top+.75, x(window.end_seconds)-x(window.start_seconds)-1.5, height-1.5);
  }
  function waveform(canvas, view, enabled, predictions, selected=0) {
    const {ctx,width,height} = context(canvas);
    if (!view) {
      canvas.setAttribute('aria-label', 'Multichannel EEG waveform. No example selected.');
      return;
    }
    const left=82, right=12, top=28, bottom=42, count=view.signal_uv[0].length;
    const start=view.view_start_seconds, duration=count/256, ids=view.channels.map((_,i)=>i).filter(i=>enabled.has(i));
    const x=t=>left+(t-start)/duration*(width-left-right), plotHeight=height-top-bottom;
    const selectedWindow=predictions?.windows[selected];
    canvas.setAttribute('aria-label', `${ids.length} EEG channels: ${ids.map(i=>view.channels[i]).join(', ')}. Recording time ${start} to ${start+duration} seconds.${selectedWindow?` Selected window ${selectedWindow.start_seconds} to ${selectedWindow.end_seconds} seconds.`:''} Display traces are centered and share an automatic gain; original samples are unchanged.`);
    ctx.save(); ctx.beginPath(); ctx.rect(left,top,width-left-right,plotHeight); ctx.clip();
    regions(ctx,view.example.seizures,x,top,plotHeight,colors.annotation);
    regions(ctx,predictions?.high_score_regions||[],x,top,plotHeight,colors.above);
    selection(ctx,selectedWindow,x,top,plotHeight);
    // Preserve the existing display gain and centering, independently of model inputs.
    const amplitude=[]; for(const i of ids){for(let j=0;j<count;j+=16)amplitude.push(Math.abs(view.signal_uv[i][j]));}
    amplitude.sort((a,b)=>a-b);const gain=Math.max(20,amplitude[Math.floor(amplitude.length*.95)]||20);
    const spacing=plotHeight/Math.max(ids.length,1);
    ids.forEach((i,k)=>{
      const base=top+(k+.5)*spacing;
      ctx.strokeStyle=colors.border; ctx.lineWidth=.5;
      ctx.beginPath();ctx.moveTo(left,base);ctx.lineTo(width-right,base);ctx.stroke();
      const samples=view.signal_uv[i];let mean=0;for(const value of samples)mean+=value/samples.length;
      ctx.strokeStyle=colors.trace;ctx.lineWidth=.8;ctx.beginPath();
      for(let j=0;j<count;j++){const px=x(start+j/256),py=base-(samples[j]-mean)/gain*spacing*.36;j?ctx.lineTo(px,py):ctx.moveTo(px,py);}ctx.stroke();
    });
    selectionBorder(ctx,selectedWindow,x,top,plotHeight);
    ctx.restore();
    regionRibbons(ctx,view.example.seizures,predictions?.high_score_regions||[],x,left,width-left-right,top);
    ctx.fillStyle=colors.text; ctx.textAlign='right';
    ids.forEach((i,k)=>ctx.fillText(view.channels[i],left-10,top+(k+.5)*spacing+4));
    ctx.fillStyle=colors.secondary; ctx.textAlign='center';
    const tickStep=width-left-right<220?8:width<480?4:2;
    for(let t=Math.ceil(start/tickStep)*tickStep;t<start+duration;t+=tickStep)ctx.fillText(`${t}`,x(t),height-22);
    ctx.font=`12px ${sans}`;ctx.fillText('Recording time (s)',left+(width-left-right)/2,height-5);
    ctx.font=`11px ${mono}`;ctx.textAlign='right';
    ctx.fillText(`Shared gain · ±${Math.round(gain)} µV`,width-right,16);
    if (!ids.length) {
      ctx.font=`14px ${sans}`;ctx.textAlign='center';
      ctx.fillText('Select channels to show their EEG traces.',left+(width-left-right)/2,top+plotHeight/2);
    }
  }
  function timeline(canvas,prediction,selected,annotations=[]) {
    // Horizontal margins match the existing timeline click-to-window mapping.
    const {ctx,width,height}=context(canvas), left=32,right=12,top=28,bottom=42;
    if(!prediction){canvas.setAttribute('aria-label','Model scores across consecutive EEG windows. No example selected.');return;}
    const windows=prediction.windows,start=windows[0].start_seconds,end=windows.at(-1).end_seconds;
    const x=t=>left+(t-start)/(end-start)*(width-left-right), y=s=>height-bottom-s*(height-top-bottom);
    const selectedWindow=windows[selected], plotHeight=height-top-bottom;
    canvas.setAttribute('aria-label', `Uncalibrated model score timeline from ${start} to ${end} seconds. Frozen threshold ${prediction.threshold.toFixed(3)}. Use the arrow keys to change windows, Home for the first window, and End for the last window.`);
    ctx.save();ctx.beginPath();ctx.rect(left,top,width-left-right,plotHeight);ctx.clip();
    regions(ctx,annotations,x,top,plotHeight,colors.annotation);
    regions(ctx,prediction.high_score_regions||[],x,top,plotHeight,colors.above);
    selection(ctx,selectedWindow,x,top,plotHeight);
    ctx.strokeStyle=colors.border;ctx.lineWidth=.5;
    for(const score of [0,.5,1]){ctx.beginPath();ctx.moveTo(left,y(score));ctx.lineTo(width-right,y(score));ctx.stroke();}
    for(let i=0;i<windows.length;i++){
      const w=windows[i];ctx.fillStyle=colors.trace;
      ctx.fillRect(x(w.start_seconds)+1,y(w.score),Math.max(1,x(w.end_seconds)-x(w.start_seconds)-2),height-bottom-y(w.score));
    }
    ctx.strokeStyle=colors.secondary;ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(left,y(prediction.threshold));ctx.lineTo(width-right,y(prediction.threshold));ctx.stroke();
    ctx.setLineDash([]);selectionBorder(ctx,selectedWindow,x,top,plotHeight);ctx.restore();
    regionRibbons(ctx,annotations,prediction.high_score_regions||[],x,left,width-left-right,top);
    ctx.font=`11px ${mono}`;ctx.fillStyle=colors.secondary;ctx.textAlign='right';
    for(const score of [0,.5,1])ctx.fillText(score.toFixed(1),left-6,y(score)+4);
    ctx.fillStyle=colors.text;ctx.fillText(`Threshold ${prediction.threshold.toFixed(3)}`,width-right,16);
    ctx.fillStyle=colors.secondary;ctx.textAlign='left';ctx.fillText(`${start}`,left,height-22);
    ctx.textAlign='right';ctx.fillText(`${end}`,width-right,height-22);
    ctx.font=`12px ${sans}`;ctx.textAlign='center';ctx.fillText('Recording time (s)',left+(width-left-right)/2,height-5);
  }
  return {waveform,timeline};
})();
