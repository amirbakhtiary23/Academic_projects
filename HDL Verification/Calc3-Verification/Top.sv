`ifndef GAURD_TESTBENCH
`define GUARD_TESTBENCH
`include "Interface.sv"
`include "Generator.sv"
`include "Monitor.sv"
`include "refrenceModel.sv"
`include "Checker.sv"
module TopProgram(
GlobalPorts globalPorts,
Stimuli stimuliPort1, Stimuli stimuliPort2,
Stimuli stimuliPort3, Stimuli stimuliPort4,
Output outputPort1, Output outputPort2,
Output outputPort3, Output outputPort4);

Monitor monitor= new (globalPorts, outputPort1,outputPort2,outputPort3,outputPort4);
Generator generator =new (globalPorts,
stimuliPort1,stimuliPort2,stimuliPort3,stimuliPort4);
referenceModel scoreboard = new (
stimuliPort1,stimuliPort2,stimuliPort3,stimuliPort4,globalPorts);
Check check= new (stimuliPort1,stimuliPort2,stimuliPort3,stimuliPort4,
 outputPort1,outputPort2,outputPort3,outputPort4,globalPorts,scoreboard);
initial begin
fork
    scoreboard.start();
    monitor.start();
    generator.start();
    check.start();
join_any
$display ("finished");
end
endmodule
`endif
