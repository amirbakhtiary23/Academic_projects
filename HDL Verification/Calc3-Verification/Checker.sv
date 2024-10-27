`include "refrenceModel.sv"
class Check;
int pointer[1:4] ;
virtual Stimuli input_ports[1:4];
virtual Output output_ports[1:4];
referenceModel scoreboard;
virtual GlobalPorts globalPort;
function new (virtual Stimuli i_port1,virtual Stimuli i_port2,virtual Stimuli i_port3,virtual Stimuli i_port4,
virtual Output o_port1,virtual Output o_port2,virtual Output o_port3,virtual Output o_port4, virtual GlobalPorts globalPorts_,
referenceModel scoreboard_ );
this.input_ports[1]=i_port1;
this.input_ports[2]=i_port2;
this.input_ports[3]=i_port3;
this.input_ports[4]=i_port4;
this.output_ports[1]=o_port1;
this.output_ports[2]=o_port2;
this.output_ports[3]=o_port3;
this.output_ports[4]=o_port4;
this.globalPort=globalPorts_;
this.scoreboard=scoreboard_;

endfunction
task start();
fork
scanner(1);
scanner (2);
scanner (3);
scanner (4);
join
endtask

task scanner(int port);
forever begin
@(this.output_ports[port].outx_tag) ;
this.pointer[port]=0;
//repeat (10) begin
//  @(posedge this.globalPort.clock);

for (int i=1 ;i<5;i++)begin
    if (this.scoreboard.outputTags[port][i]==this.output_ports[port].outx_tag)begin
    this.pointer[port]=i;
    end
        
    
if (this.pointer[port]==0)begin//this.scoreboard.outputTags[port]!=this.output_ports[port].outx_tag)begin
    $display ("[****CHECKER ERROR LOG****]  : Port ",port," Missmatch output tag");
    $display ("Output tag : ",this.output_ports[port].outx_tag,"ScoreBoard Tag : ",this.scoreboard.outputTags[port][this.pointer[port]]);
end
if (this.pointer[port]!=0)begin
if (this.scoreboard.outputResp[port][this.pointer[port]]!=this.output_ports[port].outx_resp)begin
    $display ("[****CHECKER ERROR LOG****]  : Port ",port," Missmatch output resp");
    $display ("Output resp : ",this.output_ports[port].outx_resp,"ScoreBoard resp : ",this.scoreboard.outputResp[port][this.pointer[port]]);
end 
end
if (this.scoreboard.outputPorts[port]!=this.output_ports[port].outx_data)begin
    $display ("[****CHECKER ERROR LOG****] : Port ",port," Missmatch output data");
    $display ("Output data : ",this.output_ports[port].outx_data,"ScoreBoard data : ",this.scoreboard.outputPorts[port]);
end
end
end
endtask


endclass
