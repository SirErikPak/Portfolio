USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.views WHERE object_id = OBJECT_ID(N'[dbo].[vw_ServerStorage]'))
DROP VIEW [dbo].[vw_ServerStorage]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[vw_ServerStorage]
As
/*********************************************************************/
/* Database: WyethDBA  											     */
/* Purpose: Prep EMAIL for Storage Failure						     */
/* Note: 	This View is Used by P_Monitor_ScheduleJobFailure Proc	 */
/*			to Generate Storage EMAIL							     */
/*                                                                   */
/*  Org. Date: 09/06/2006        EPAK                                */
/*  Mod. Date: 00/00/0000                                            */
/*********************************************************************/
Select	DI.JobDateInfo As LastProcessedDate,
		SER.ServerID,
		SER.SiteID,
		DST.ServerTypeDesc,
		SIT.SiteDesc,
		SER.ServerName,
		DSD.DriveLetter,
		DSD.TotalDiskSpace_MB,
		DSD.FreeDiskSpace_MB,
		DSD.FreeDiskPercentage * 100 As FreeDiskPercentage
From	DatabaseServerDiskInfo DSD INNER JOIN DateInfo DI
ON		DI.DateID = DSD.DateID
		INNER JOIN DatabaseServer SER
ON		DSD.ServerID = SER.ServerID
		INNER JOIN DatabaseSite SIT
ON		SER.SiteID = SIT.SiteID
		INNER JOIN DatabaseServerType DST
ON		SER.ServerTypeID = DST.ServerTypeID
		INNER JOIN	DatabaseDiskSizeThreshold DDST
ON		DSD.DiskSizeThresholdID = DDST.DiskSizeThresholdID
Where	DDST.DiskSizeThresholdPercent/100.00 >= DSD.FreeDiskPercentage
And		DI.DateID = (Select MAX(DateID) From DateInfo)
And		DI.ActiveFlag = 'Y'

/*********************************************************************/
/*								                                     */
/*********************************************************************/
GO
GRANT  SELECT  ON [dbo].[vw_ServerStorage]  TO [WyethDBA_Client]