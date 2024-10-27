`ifndef GUARD_DRIVER
`define GUARD_DRIVER
`include "Interface.sv"

class Driver; 
int unsigned pNumber;//declaring port X
virtual Stimuli port;
virtual GlobalPorts globalPorts;

//declaring stimuli for each port on instantiating driver class
function new( int unsigned _pNumber, virtual GlobalPorts _globalPorts, 
virtual Stimuli _port);
begin 
this.pNumber=_pNumber;
this.port = _port;
this.globalPorts = _globalPorts;
end
endfunction

//defining add task 
task add (logic [0:3] input1, logic [0:3] input2,logic [0:3] input3,logic [0:1] input4);
begin 
this.port.reqx_cmd = 4'b0001;
this.port.reqx_d1=input1;
this.port.reqx_d2=input2;
this.port.reqx_r1=input3;
this.port.reqx_tag=input4;
this.port.reqx_data=0;
@(posedge globalPorts.clock);
this.port.reqx_cmd = 0;
this.port.reqx_d1=0;
this.port.reqx_r1=0;
//this.port.reqx_tag=0;
end
endtask

task sub (logic [0:3] input1, logic [0:3] input2,logic [0:3] input3,logic [0:1] input4);
begin 
this.port.reqx_cmd = 4'b0010;
this.port.reqx_d1=input1;
this.port.reqx_d2=input2;
this.port.reqx_r1=input3;
this.port.reqx_tag=input4;
this.port.reqx_data=0;
@(posedge globalPorts.clock);
this.port.reqx_cmd = 0;
this.port.reqx_d1=0;
this.port.reqx_r1=0;
//this.port.reqx_tag=0;
end
endtask


task shiftLeft (logic [0:3] input1, logic [0:3] input2,logic [0:3] input3,logic [0:1] input4);
begin 
this.port.reqx_cmd = 4'b0101;
this.port.reqx_d1=input1;
this.port.reqx_d2=input2;
this.port.reqx_r1=input3;
this.port.reqx_tag=input4;
this.port.reqx_data=0;
@(posedge globalPorts.clock);
this.port.reqx_cmd = 0;
this.port.reqx_d1=0;
this.port.reqx_r1=0;
//this.port.reqx_tag=0;
end
endtask

task shiftRight (logic [0:3] input1, logic [0:3] input2,logic [0:3] input3, logic [0:1] input4);
begin 
this.port.reqx_cmd = 4'b0110;
this.port.reqx_d1=input1;
this.port.reqx_d2=input2;
this.port.reqx_r1=input3;
this.port.reqx_tag=input4;
this.port.reqx_data=0;
@(posedge globalPorts.clock);
this.port.reqx_cmd = 0;
this.port.reqx_d1=0;
this.port.reqx_r1=0;
//this.port.reqx_tag=0;
end
endtask

task store (logic [0:31] input1, logic [0:3] input2, logic [0:1] input3);
begin
this.port.reqx_cmd=4'b1001;
this.port.reqx_data=input1;
this.port.reqx_r1=input2;
this.port.reqx_tag=input3;
this.port.reqx_d1=0;
this.port.reqx_d2=0;
@ (posedge globalPorts.clock);
this.port.reqx_data=0;
this.port.reqx_r1=0;
//this.port.reqx_tag=0;
end
endtask

task fetch (logic [0:3] input1,  logic [0:1] input2);
begin
this.port.reqx_cmd=4'b1010;
this.port.reqx_d1=input1;
this.port.reqx_tag=input2;
this.port.reqx_data=0;
this.port.reqx_r1=0;
this.port.reqx_d2=0;
@(posedge globalPorts.clock);
this.port.reqx_d1=0;
//this.port.reqx_tag=0;
end
endtask

task beqz (logic [0:3] input1, logic [0:1] input2);
begin
this.port.reqx_cmd=4'b1100;
this.port.reqx_d1=input1;
this.port.reqx_tag=input2;
this.port.reqx_d2=0;
this.port.reqx_data=0;
this.port.reqx_r1=0;
@(posedge globalPorts.clock);
//this.port.reqx_tag=0;
this.port.reqx_d1=0;
end
endtask

task beq (logic [0:3] input1,logic [0:3] input2, logic [0:1] input3);
begin
this.port.reqx_cmd=4'b1101;
this.port.reqx_d1=input1;
this.port.reqx_tag=input3;
this.port.reqx_d2=input2;
this.port.reqx_data=0;
this.port.reqx_r1=0;
@(posedge globalPorts.clock);
//this.port.reqx_tag=0;
this.port.reqx_d1=0;
this.port.reqx_d2=0;
end
endtask

endclass
`endif
