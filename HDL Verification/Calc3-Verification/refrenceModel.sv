`ifndef REFERENCEMODEL
`define REFERENCEMODEL
class referenceModel;
virtual Stimuli ports[1:4];
virtual GlobalPorts globalPorts;
int pointer[1:4] ;
bit [0:15][0:31]registers;
bit [1:4][0:1] inputTags;
bit [1:4][0:31] outputPorts;
bit [1:4][1:4][0:1] outputTags;
bit [1:4][1:4][0:1] outputResp;
bit [0:31] overflowDetector;
bit [0:32] overflowTemp;
bit [0:31] underflowDetector;
bit [0:32] underflowTemp;
function new(virtual Stimuli _port1,virtual Stimuli _port2,virtual Stimuli _port3,virtual Stimuli _port4, virtual GlobalPorts globalPorts_);
begin
    this.globalPorts=globalPorts_;
    this.ports[1]=_port1;
    this.ports[2]=_port2;
    this.ports[3]=_port3;
    this.ports[4]=_port4;

end

endfunction
task start();
$display("Scoreboard Running.");
fork
    scan(1);
    scan(2);
    scan(3);
    scan(4);

join
endtask
function int tagSelector(int port, bit tag);
begin
if (outputTags[port][1]==2'bz) begin
outputTags[port][1]=tag;
this.pointer[port]=1;
return 1;
end
if (outputTags[port][2]==2'bz) begin
outputTags[port][2]=tag;
this.pointer[port]=2;
return 2;
end
if (outputTags[port][3]==2'bz) begin
outputTags[port][3]=tag;
this.pointer[port]=3;
return 3;
end
if (outputTags[port][4]==2'bz) begin
outputTags[port][4]=tag;
this.pointer[port]=4;
return 4;
end
outputTags[port][1]=tag;
outputTags[port][2]=2'bz;
outputTags[port][3]=2'bz;
outputTags[port][4]=2'bz;
this.pointer[port]=1;
return 1;
end
endfunction

task scan(int port);
forever fork
@this.ports[port].reqx_tag;
if (this.ports[port].reqx_cmd == 4'b0001) begin
    add(port,this.ports[port].reqx_tag,this.ports[port].reqx_d1,this.ports[port].reqx_d2,this.ports[port].reqx_r1);
end

if (this.ports[port].reqx_cmd == 4'b0010) begin
    sub(port,this.ports[port].reqx_tag,this.ports[port].reqx_d1,this.ports[port].reqx_d2,this.ports[port].reqx_r1);
end

if (this.ports[port].reqx_cmd == 4'b0101) begin
    shiftLeft(port,this.ports[port].reqx_tag,this.ports[port].reqx_d1,this.ports[port].reqx_d2,this.ports[port].reqx_r1);
end

if (this.ports[port].reqx_cmd == 4'b0110) begin
    shiftRight(port,this.ports[port].reqx_tag,this.ports[port].reqx_d1,this.ports[port].reqx_d2,this.ports[port].reqx_r1);
end
if (this.ports[port].reqx_cmd == 4'b1001) begin
    store(port,this.ports[port].reqx_tag,this.ports[port].reqx_r1,this.ports[port].reqx_data);
    $display ("Storing to scoreboard register :", this.ports[port].reqx_r1," data : ",this.ports[port].reqx_data,"Tag : ",this.ports[port].reqx_tag);

end
if (this.ports[port].reqx_cmd == 4'b1010) begin
    fetch(port,this.ports[port].reqx_tag,this.ports[port].reqx_d1);
    $display ("Fetching from scoreboard :", this.registers[this.ports[port].reqx_d1],"Tag : ",this.ports[port].reqx_tag);
end



join
endtask

task add(input int port, input [0:1] tag,input bit [0:3] input1 ,input bit [0:3] input2, input bit [0:3] input3);
this.registers[input3]=this.registers[input1]+this.registers[input2];
this.overflowTemp=this.registers[input1]+this.registers[input2];
this.tagSelector(port,tag);
if (this.overflowTemp>this.overflowDetector) begin
this.outputResp[port][this.pointer[port]]=2;
end
else begin
this.outputResp[port][this.pointer[port]]=1;
end
endtask

task sub(input int port, input [0:1] tag, input bit [0:3] input1, input bit [0:3] input2, input bit [0:3] input3);
this.registers[input3]=this.registers[input2]-this.registers[input1];
this.tagSelector(port,tag);
if (this.registers[input1]>this.registers[input2]) begin
this.outputResp[port][this.pointer[port]]=2;
end
else begin
this.outputResp[port][this.pointer[port]]=1;
end
endtask

task store(input int port, input bit[0:1] tag, input bit [0:3] input1, input bit [0:31] input2);
this.registers[input1]=input2;
this.tagSelector(port,tag);
this.outputResp[port][this.pointer[port]]=1;
endtask;

task fetch(input int port, input bit [0:1] tag, input bit [0:3] input1);
this.tagSelector(port,tag);
this.outputPorts[port]=this.registers[input1];
this.outputResp[port][this.pointer[port]]=1;
endtask

task shiftLeft (input int port, input [0:1] tag, input bit [0:3] input1, input bit [0:3] input2, input bit [0:3] input3);
this.tagSelector(port,tag);
this.registers[input3]=this.registers[input1]<<this.registers[input2][27:31];
this.outputResp[port][this.pointer[port]]=1;
endtask

task shiftRight (input int port, input [0:1] tag, input bit [0:3] input1, input bit [0:3] input2, input bit [0:3] input3);
this.tagSelector(port,tag);
this.registers[input3]=this.registers[input1]>>this.registers[input2][27:31];
this.outputResp[port][this.pointer[port]]=1;
endtask
endclass
`endif
/*
task branch ();


endclass
module show ();
//referenceArrays arr;
referenceModel model;
//bit [0:31] temp;
initial begin
model=new();
model.overflowDetector = 32'hffffffff;
//arr=new();
model.registers[5]=32'h00000002;

$display ("%0b",model.registers[5]);
model.registers[6]=32'h00000001;
//$display ("%0d",arr.registers[6]);
model.shiftRight(1,2,5,6,2);
//arr.registers[2]=temp;
//$display ("%0d",temp);
$display ("%0b",model.registers[2]);
$display ("%0b",model.outputResp[1]);
end
endmodule*/