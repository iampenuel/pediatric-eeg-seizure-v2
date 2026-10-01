/* Small, bundled canvas plotting module. Display transforms never affect inference. */
window.EEGPlot = (() => {
  function context(canvas) {
    const dpr = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    canvas.width = Math.round(box.width * dpr); canvas.height = Math.round(box.height * dpr);
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr); ctx.font = '10px -apple-system, sans-serif';
    return {ctx, width:box.width, height:box.height};
  }
  function regions(ctx, intervals, x, top, height, color) {
    ctx.fillStyle=color; for (const [a,b] of intervals) {const left=x(a),right=x(b);ctx.fillRect(left,top,right-left,height);}
  }
  function waveform(canvas, view, enabled, predictions) {
    const {ctx,width,height}=context(canvas); if(!view)return;
    const left=62,right=12,top=12,bottom=25, count=view.signal_uv[0].length;
    const start=view.view_start_seconds, duration=count/256, ids=view.channels.map((_,i)=>i).filter(i=>enabled.has(i));
    const x=t=>left+(t-start)/duration*(width-left-right);
    ctx.save();ctx.beginPath();ctx.rect(left,top,width-left-right,height-top-bottom);ctx.clip();
    regions(ctx,view.example.seizures,x,top,height-top-bottom,'#dce9de');
    regions(ctx,predictions?.high_score_regions||[],x,top,height-top-bottom,'#e7b56435');
    const amplitude=[]; for(const i of ids){for(let j=0;j<count;j+=16)amplitude.push(Math.abs(view.signal_uv[i][j]));}
    amplitude.sort((a,b)=>a-b);const gain=Math.max(20,amplitude[Math.floor(amplitude.length*.95)]||20);
    const spacing=(height-top-bottom)/Math.max(ids.length,1);
    ids.forEach((i,k)=>{const base=top+(k+.5)*spacing;ctx.strokeStyle='#bacabe66';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(left,base);ctx.lineTo(width-right,base);ctx.stroke();
      const samples=view.signal_uv[i];let mean=0;for(const value of samples)mean+=value/samples.length;
      ctx.strokeStyle='#265a5a';ctx.lineWidth=.7;ctx.beginPath();for(let j=0;j<count;j++){const px=x(start+j/256),py=base-(samples[j]-mean)/gain*spacing*.36;j?ctx.lineTo(px,py):ctx.moveTo(px,py);}ctx.stroke();});
    ctx.restore();ctx.fillStyle='#607272';ids.forEach((i,k)=>ctx.fillText(view.channels[i],0,top+(k+.5)*spacing+3));
    for(let t=Math.ceil(start/2)*2;t<start+duration;t+=2){ctx.fillText(`${t}s`,x(t)-10,height-6);}
    ctx.fillText(`±${Math.round(gain)} µV`,Math.max(left,width-88),11);
  }
  function timeline(canvas,prediction,selected,annotations=[]) {
    const {ctx,width,height}=context(canvas), left=32,right=12,top=18,bottom=25;
    ctx.fillStyle='#607272';ctx.fillText('1.0',5,top+3);ctx.fillText('0.0',5,height-bottom+3);
    if(!prediction)return;
    const windows=prediction.windows,start=windows[0].start_seconds,end=windows.at(-1).end_seconds;
    const x=t=>left+(t-start)/(end-start)*(width-left-right), y=s=>height-bottom-s*(height-top-bottom);
    ctx.save();ctx.beginPath();ctx.rect(left,top,width-left-right,height-top-bottom);ctx.clip();
    regions(ctx,annotations,x,top,height-top-bottom,'#dbe7dc');
    for(let i=0;i<windows.length;i++){const w=windows[i];ctx.fillStyle=i===selected?'#207974':'#79a39a';ctx.fillRect(x(w.start_seconds)+1,y(w.score),Math.max(1,x(w.end_seconds)-x(w.start_seconds)-2),height-bottom-y(w.score));}
    ctx.strokeStyle='#b47736';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(left,y(prediction.threshold));ctx.lineTo(width-right,y(prediction.threshold));ctx.stroke();ctx.restore();
    ctx.fillStyle='#607272';ctx.fillText(`${start}s`,left,height-6);ctx.fillText(`${end}s`,width-45,height-6);
  }
  return {waveform,timeline};
})();
