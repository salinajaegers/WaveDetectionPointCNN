extensions [csv]
breed [ cells cell ]

cells-own [
  hex-neighbors           ;; agentset of 6 neighboring cells
  n1                      ;; used to store a count
  n2                      ;; used to store a count
  n3                      ;; used to store a count
  n4                      ;; used to store a count
  countdownERK            ;; to set the count down of ERK activity pulse duration
  countdownSurvival       ;; to set the count down of the survival time
  countdownApoptosisColor ;; count-down of the apoptosis color (only viaual)
  countdownReplacement    ;; time that an apoptotic cell needs to be replaced by another one that can be restimulated
  RowNumber               ;; the row number in each specific moment
  JustActivated           ;; if 1 or 2 it means that the cell has just been activated and cannot be activated again
  FileName
  WaveID                  ;; inherits the cell ID from the source cell (apoptotic or not)

  ;; SETTINGS
  MaxNumberRows           ;; parameter that determine the Max number of rows that it propagates
  ERKPulseDuration        ;; parameter that tells how long a ERk pulse lasts (5 minutes time interval)
  SurvivalDuration        ;; parameter that tells how long the survival lasts (5 minutes time interval)
  ApoptosisColorDuration  ;; duration of apoptosis red color (only for visualization, in 5 minutes interval)
  ReplacementDuration     ;; time that is needed for an apoptotic cell to be replaced (5 minutes time interval)
  SurvivalProbability     ;; parameter that tells the probability to survive of a cell in survival mode (expressed in percentage)
  PropagationError        ;; percentage of cells that fail to propagate the wave
  ApoptoticRate           ;; apoptotic rate per 5 minutes (in 10,000)
  RefractoryTime          ;; the time in which a cell cannot be re-activated. Important for directional propagation of the waves. Typically the duration of the ERK pulse (in 5 minutes)
  MinActiveNeighbors      ;; the minimum number of active neighbors that is required to propagate the wave
  NoApoSourceProbability  ;; NoApoSource rate per 5 minutes (in 10,000)

  ;; MEASURES FOR POST ANALYSIS
  Apoptosis               ;; if 1 is apoptosis, if 0 no apoptosis
  ERKactivity             ;; if 1 ERK is on
  Survival                ;; if 1 survival is on
  ERKactivityValue        ;; ERK activity value to simulate a pulse
  NoApoSource]            ;; if it is 1 it is a no apo source

globals [ExportData
         ExportImages]

to setup
  clear-all
  setup-grid
  reset-ticks
end

to setup-grid
  set-default-shape cells "hex"
  ask patches
    [ sprout-cells 1
        [ set color gray - 2  ;; dark gray
          ;; shift even columns down
          if pxcor mod 2 = 0
            [ set ycor ycor - 0.5 ] ] ]
  ;; now set up the hex-neighbors agentsets
  ask cells
    [ ifelse pxcor mod 2 = 0
        [ set hex-neighbors cells-on patches at-points [[0 1] [1 0] [1 -1] [0 -1] [-1 -1] [-1 0]] ]
        [ set hex-neighbors cells-on patches at-points [[0 1] [1 1] [1  0] [0 -1] [-1  0] [-1 1]] ] ]
  ;; now set up parameters that have to start from 0
  ask cells
    [set countdownERK 0
     set countdownSurvival 0
     set countdownApoptosisColor 0
     set countdownReplacement 0
     set RowNumber 0
     set NoApoSource 0
     set WaveID -1]

  ;; NOW SET UP THE SETTINGS [THIS IS THE ONLY PART THAT IS IMPORTANT TO YOU, MACIEJ]
    ask cells
    [set MaxNumberRows 3          ;; parameter that determine the Max number of rows that it propagates (0-infinite)
     set ERKPulseDuration 7        ;; parameter that tells how long a ERk pulse lasts (5 minutes time interval) (0-infinite)
     set SurvivalDuration 24       ;; parameter that tells how long the survival lasts (5 minutes time interval) (0-infinite)
     set ApoptosisColorDuration 7  ;; duration of apoptosis red color (only for visualization, in 5 minutes interval) (0-infinite)
     set ReplacementDuration 0     ;; time that is needed for an apoptotic cell to be replaced (5 minutes time interval) (0-infinite)
     set SurvivalProbability 70    ;; parameter that tells the probability to survive of a cell in survival mode (expressed in percentage) (0-100)
     set PropagationError 30       ;; percentage of cells that fail to propagate the wave (0-100)
     set ApoptoticRate 2           ;; apoptotic rate per 5 minutes (in 10,000) (0-10,000)
     set NoApoSourceProbability 0  ;; NoApoSource rate per 5 minutes (in 10,000) (0-10,000)
     set RefractoryTime 7          ;; the time in which a cell cannot be re-activated. Important for directional propagation of the waves. Typically the duration of the ERK pulse (in 5 minutes) (0-infinite)
     set MinActiveNeighbors 1]     ;; the minimum number of active neighbors that is required to propagate the wave (1-3)
     set ExportData 1              ;; 1 to export, 0 to not export. The columns are: ticks who xcor ycor apoptosis ERKactivity Survival ERKactivityValue WaveID (0-1)
     set ExportImages 0            ;; 1 to export, 0 to not export (0-1)

  end


to go

  ;; activate the apoptotic cell or the non apoptotic source to communicate the wave
  ask cells
    [ if apoptosis = 1
       [ set RowNumber 1
         set countdownERK ERKPulseDuration + 1
         set countdownSurvival SurvivalDuration + 1
         set countdownApoptosisColor ApoptosisColorDuration + 1
         set countdownReplacement ReplacementDuration + 1
         set apoptosis 0
         set JustActivated RefractoryTime
         set WaveID who] ]
    ask cells
    [ if NoApoSource = 1
       [ set RowNumber 1
         set countdownERK ERKPulseDuration + 1
         set countdownSurvival SurvivalDuration + 1
         set NoApoSource 0
         set ERKactivity 1
         set Survival 1
         set JustActivated RefractoryTime
         set WaveID who] ]

  ;; propagate the wave
  ask cells
    [ set n1 count hex-neighbors with [RowNumber >= 1] ]
  ask cells
    [ set n2 count hex-neighbors with [JustActivated = RefractoryTime - 1] ]
  ask cells
    [ set n3 max [RowNumber] of hex-neighbors ]
  ask cells
    [ set n4 [WaveID] of max-one-of hex-neighbors [RowNumber] ]
  ask cells
    [ if n1 >= MinActiveNeighbors and n2 > 0 and JustActivated <= 0 and countdownReplacement = 0
        [ set RowNumber n3 + 1
          set countdownERK ERKPulseDuration + 1
          set countdownSurvival SurvivalDuration + 1
          set ERKactivity 1
          set Survival 1
          set JustActivated RefractoryTime
          set WaveID n4] ]
  ask cells
     [ if MaxNumberRows = 0
        [set ERKactivity 0
         set Survival 0
         set JustActivated 0
         set WaveID -1] ]

  ;; add losses to the propagation of the wave
  ask cells
    [ if random 100 < PropagationError and JustActivated = RefractoryTime
        [ set JustActivated RefractoryTime - 1] ]

  ;; Reduce the countdowns of 1 every tick and cells that finish the countadown become grey
  ask cells
    [set countdownSurvival countdownSurvival - 1]
  ask cells
    [set countdownERK countdownERK - 1]
  ask cells
    [set countdownApoptosisColor countdownApoptosisColor - 1]
  ask cells
    [set countdownReplacement countdownReplacement - 1]
  ask cells
    [set JustActivated JustActivated - 1]

  ;; Brig negative numbers to zero and reset colors
  ask cells
    [ if countdownERK <= 0
       [ set countdownERK 0
         set ERKactivity 0
         set ERKactivityValue 0
         set WaveID -1 ] ]
  ask cells
    [ if countdownSurvival <= 0
       [ set countdownSurvival 0
         set Survival 0 ] ]
  ask cells
    [ if countdownApoptosisColor <= 0
      [ set countdownApoptosisColor 0 ] ]
  ask cells
    [ if countdownReplacement <= 0
      [ set countdownReplacement 0 ] ]
  ask cells
    [ if JustActivated < 0
       [ set JustActivated 0 ] ]

  ;; Resets the NumberRows on the base of the Just Activated
    ask cells
    [ if JustActivated <= 0
       [ set RowNumber 0 ] ]

  ;; bring the row counter to 0
    ask cells
    [ if RowNumber = MaxNumberRows + 1
       [ set RowNumber 0 ] ]

  ;; Only Survival 0 cells can undergo 100% apoptosis, white survival 1 have lower chance
  ;; Non apoptotic source
  ask cells
    [ if random 10000 < ApoptoticRate and countdownReplacement = 0
        [ set apoptosis 1] ]
  ask cells
    [ if random 100 <= SurvivalProbability and Survival = 1 and Apoptosis = 1
        [ set apoptosis 0] ]
  ask cells
    [ if random 10000 < NoApoSourceProbability and countdownReplacement = 0
        [ set NoApoSource 1] ]

  ;; Colors ERK gradient
    ask cells
    [ if countdownERK = ERKPulseDuration and ERKactivity = 1
      [ set color white - 4
        set ERKactivityValue 2 ]]
  ask cells
    [ if countdownERK = ERKPulseDuration - 1 and ERKactivity = 1
      [ set color white
        set ERKactivityValue 10]]
  ask cells
    [ if countdownERK = ERKPulseDuration - 2 and ERKactivity = 1
      [ set color white
        set ERKactivityValue 8]]
  ask cells
    [ if countdownERK = ERKPulseDuration - 3 and ERKactivity = 1
      [ set color white - 1
        set ERKactivityValue 5]]
  ask cells
    [ if countdownERK = ERKPulseDuration - 4 and ERKactivity = 1
      [ set color white - 3
        set ERKactivityValue 3]]
  ask cells
    [ if countdownERK = ERKPulseDuration - 5 and ERKactivity = 1
      [ set color white - 4
        set ERKactivityValue 2]]
  ask cells
    [ if countdownERK = ERKPulseDuration - 6 and ERKactivity = 1
      [ set color white - 5
        set ERKactivityValue 1]]
  ask cells
    [ if countdownERK = 0 and Survival = 0 and ERKactivity = 0
      [ set ERKactivityValue 0]]

    ;; Other Colors
;  ask cells
;    [ if apoptosis = 1
;        [ set color red] ]
;  ask cells
;    [ if countdownApoptosisColor >= 1
;       [ set color red ] ]
  ;;ask cells
    ;;[ if countdownApoptosisColor >= 1
       ;;[ set color red ] ]
;  ask cells
;    [ if ERKactivity = 0 and countdownERK = 0 and Survival = 0
;       [ set color grey - 2 ] ]
    ask cells
    [ if ERKactivity = 0 and countdownERK = 0
       [ set color grey - 2 ] ]
;  ask cells
;    [ if ERKactivity = 0 and Survival = 1
;       [ set color blue + 2  ] ]
  ask cells
    [ if countdownReplacement > 0 and countdownApoptosisColor = 0
       [ set color black ] ]

  ;; Export csv and picture
  if ExportData = 1 [csv:to-file word ticks ".csv" [ (list ticks who xcor ycor apoptosis ERKactivity Survival ERKactivityValue WaveID) ] of turtles]
  if ExportImages = 1 [export-view word ticks ".png"]

  tick

end



; Public Domain:
; To the extent possible under law, Uri Wilensky has waived all
; copyright and related or neighboring rights to this model.
@#$#@#$#@
GRAPHICS-WINDOW
240
10
620
391
-1
-1
12.0
1
10
1
1
1
0
0
0
1
-15
15
-15
15
1
1
1
ticks
30.0

BUTTON
32
44
113
77
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
73
114
154
147
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
73
79
154
112
step
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

@#$#@#$#@
## WHAT IS IT?

Simulator of Apoptotic ERK and Survival waves in an epithelial monolayer

ERK activity in white
Apoptosis in red
Survival in blue
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

hex
false
0
Polygon -7500403 true true 0 150 75 30 225 30 300 150 225 270 75 270

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.3.0
@#$#@#$#@
setup1
repeat 20 [ go ]
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>count turtles</metric>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
1
@#$#@#$#@
